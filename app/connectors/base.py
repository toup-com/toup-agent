"""
T1c — `BaseConnectorProvider` ABC and connector result/context types.

Every connector subclasses `BaseConnectorProvider` in its `provider.py`.
The registry instantiates one provider per connector at boot and reuses
it across requests (providers are stateless — per-call user context is
in `ConnectorContext`).

The result types are a sum type per `01-architecture.md` §2.5. The
dispatcher (T1e) inspects which subclass came back and serialises it
to a structured tool-result string for the LLM.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Optional


# ─── Per-call context ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ConnectorContext:
    """Per-call context handed to provider methods.

    `vault` is intentionally typed as Any to avoid a circular import
    (the vault module imports models, the dispatcher imports both).
    Real type at runtime is `app.services.connector_vault` module.
    """

    user_id: str
    channel: str = "web"        # "web" | "telegram" | "voice" | "whatsapp" | "mobile"
    request_id: str = "no-id"   # AgentRunner request id for trace correlation
    vault: Any = None           # Injected by dispatcher; provider uses to read tokens
    metadata: dict = field(default_factory=dict)


# ─── Result sum type ──────────────────────────────────────────────────


class ConnectorResult:
    """Sum-type root. Subclasses below are the only legal members.
    Use `isinstance` checks in the dispatcher; do not construct
    `ConnectorResult` directly."""


@dataclass(frozen=True)
class ConnectorOk(ConnectorResult):
    """Normal path. `content` is whatever the provider wants the LLM to
    see — typically a JSON-encoded string for structured data."""
    content: str


@dataclass(frozen=True)
class ConnectorRateLimited(ConnectorResult):
    """Provider returned 429. Dispatcher surfaces the retry budget to
    the LLM so it can suggest "try again in N seconds"."""
    retry_after_s: int


@dataclass(frozen=True)
class ConnectorReauthRequired(ConnectorResult):
    """Token revoked at the provider OR refresh failed. Dispatcher
    flips identity status, returns a chip for the frontend with the
    reauth_url."""
    reauth_url: str


@dataclass(frozen=True)
class ConnectorProviderDown(ConnectorResult):
    """Provider returned 5xx or network unreachable. Health probe will
    flip the card to 'down' on the next sweep."""
    provider_status_url: Optional[str] = None


@dataclass(frozen=True)
class ConnectorScopeMissing(ConnectorResult):
    """User authorized us with a scope subset that doesn't include
    what this tool needs. Dispatcher tells the LLM; user must re-auth
    to grant the missing scope."""
    required_scope: str


@dataclass(frozen=True)
class ConnectorToolError(ConnectorResult):
    """Provider returned 4xx (other than 401/403/429) — caller error,
    not a connector-layer problem. `retryable` tells the dispatcher
    whether automatic retry is sensible."""
    message: str
    retryable: bool = False


# ─── Refresh / health probe results ───────────────────────────────────


@dataclass(frozen=True)
class RefreshResult:
    """What `provider.refresh()` returns to the dispatcher.

    `refresh_token` is None when the provider didn't issue a new one —
    callers should reuse the existing one.
    """

    access_token: str
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None


class RefreshFailed(Exception):
    """Raised by `provider.refresh()` when refresh is permanently
    failed (token revoked, refresh_token expired, scope insufficient).
    Dispatcher converts to ConnectorReauthRequired and flips the
    identity to `reauth_required`."""


@dataclass(frozen=True)
class HealthResult:
    """Returned by `provider.health_probe()`. `ok=True` updates the
    last-success timestamp; `ok=False` records a failure event with
    `detail` going into the audit metadata."""

    ok: bool
    detail: Optional[str] = None


# ─── Provider ABC ─────────────────────────────────────────────────────


class BaseConnectorProvider(abc.ABC):
    """Abstract base every connector's `provider.py` subclasses.

    Subclasses MUST set `manifest_id: ClassVar[str]` matching the
    `id` field in their `manifest.yaml`. The registry validates this
    at load time and rejects mismatches.
    """

    # Set on every subclass; cross-checked against manifest.yaml.id
    manifest_id: ClassVar[str]

    @abc.abstractmethod
    async def execute(
        self,
        tool_name: str,
        tool_input: dict,
        ctx: ConnectorContext,
    ) -> ConnectorResult:
        """Run a tool call. Always returns a ConnectorResult subclass —
        do not raise for provider-side errors; serialise them via the
        sum type so the dispatcher can audit consistently."""
        ...

    @abc.abstractmethod
    async def revoke(
        self,
        user_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
    ) -> None:
        """Provider-side revocation hook. Called by /oauth/disconnect
        before the vault zeroes the ciphertext. The dispatcher passes
        the decrypted access_token (and refresh_token when present)
        because most providers need it as the revoke request body.

        Best-effort: providers sometimes 404 a revoke request for an
        already-revoked token — treat that as success and move on. The
        OAuth API audits success vs failure separately so ops can grep
        revocation_provider_failed events."""
        ...

    @abc.abstractmethod
    async def refresh(self, refresh_token: str) -> RefreshResult:
        """Exchange a refresh_token for a fresh access_token.
        Raise RefreshFailed if the refresh is unrecoverable."""
        ...

    @abc.abstractmethod
    async def health_probe(self, ctx: ConnectorContext) -> HealthResult:
        """Cheap call used by the health-probe scheduler (T5c) to drive
        the card-state badge on /agent/integrations. Should be the
        smallest authenticated request the provider exposes (e.g.
        Gmail's `users.getProfile` rather than a full message search)."""
        ...
