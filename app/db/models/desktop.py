"""Toup for Mac — the desktop relay's tables.

All four are PLATFORM_ONLY. The contract they implement is
`desktop/docs/RELAY_PROTOCOL.md`; the reasoning behind each divergence from
the Chrome-extension precedent is `desktop/docs/discovery/agent-tool-relay.md`.

WHICH DATABASE, AND WHY IT IS NOT THE SAME ANSWER AS `extension_devices`
───────────────────────────────────────────────────────────────────────────
`extension_devices` is SHARED (`base.py`), and the note there gives the
reason: `app/api/extension.py` touches the model 28 times and is mounted by
BOTH mains, so the table is genuinely bi-resident and categorising it to one
side would repeat the `agent_configs` outage.

`app/api/desktop.py` is also mounted by both mains — but it is built so that
**no agent-mode code path touches these models at all**. The agent half of
that module is the WebSocket and two internal routes, and each one reaches
the platform over HTTP (`X-Agent-Key`) instead of querying locally. That is
not an accident of layout, it is the point:

  * A single store means a single truth. `agent-tool-relay.md` §9.8 and risk
    #7 are both the same defect — two stores for one policy, one writer, two
    readers, and a UI that says "blocked"/"revoked" over automation that is
    not. A SHARED `desktop_devices` would give every tenant an empty second
    copy of the revocation list, and the first connect-time check written
    against the local copy would pass every revoked device on earth while
    looking exactly like a working check.
  * The credential these rows gate can read files and run commands, so it
    belongs beside the connector tokens (`PLATFORM_ONLY`, "connector tokens
    stay platform-side, full stop") rather than in a table a tenant
    container can write.

Because they are platform-only and new, `Base.metadata.create_all` (scoped to
the allowed set in `init_db`) creates them on the platform and never on a
tenant, and they need NO entry in `database.py`'s `_alter_statements` — that
list exists so tenant DBs, which never run alembic, can self-heal columns of
tables they DO carry. There is no tenant copy to heal. They also need no
`SHARED_COLUMN_AUTHORITY` entries, since that map covers SHARED tables only.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.models.base import Base

# ─── Pairing-record lifecycle ────────────────────────────────────────────
#
# The extension keeps pending pairings in a process-local dict with a 10-min
# TTL (`extension.py:59-61`), and says so in its own module docstring: "this
# is sufficient until we scale beyond one platform replica". The platform
# runs `numReplicas: 2` (`railway.json`). So a `pair/init` served by replica A
# and a `pair/approve` served by replica B do not see each other, and pairing
# fails roughly half the time with no error that names the cause.
#
# These rows are the fix. Same 10-minute lifetime, expressed as a column so
# any replica can honour it, and `status` is a column so `poll` can tell
# "waiting" from "the user said no" from "expired" — three outcomes the
# protocol gives three different HTTP answers (§1.2).
#
# `expired` is its OWN value and not a flavour of `denied`. The lazy sweep
# runs on `pair/init`, `pair/lookup` and `pair/approve`, so a code that ran
# out is routinely rewritten by somebody else's request before its own
# device polls again — and folding it into `denied` made that device tell
# its user "you declined this on the website" about a code they never saw a
# page for. `denied` is a person's decision; `expired` is a clock.
PAIRING_STATUSES = ("pending", "approved", "denied", "expired", "consumed")

# ─── Pending-action lifecycle ────────────────────────────────────────────
#
# Byte-identical vocabulary to `connectors.PENDING_ACTION_STATUSES`, and the
# same reasoning applies verbatim: `approved` is NOT success, it means "the
# user said yes and we committed to running it", so a process death between
# the claim and the device dispatch leaves a row that can never be claimed
# twice. `dispatched` is the one addition, and it exists because this flow has
# a hop the connector flow does not have — see DesktopPendingAction below.
DESKTOP_ACTION_STATUSES = (
    "pending", "approved", "dispatched", "executed", "failed",
    "rejected", "expired",
)

DESKTOP_ACTION_TERMINAL = frozenset({
    "approved", "dispatched", "executed", "failed", "rejected", "expired",
})


class DesktopDevice(Base):
    """One Mac paired to a Toup account.

    The row is the revocation list. `token_jti` is checked on EVERY relay
    connect — which is the follow-up `extension.py:841-848` named and never
    shipped, and the single most important thing not to reproduce here
    (`agent-tool-relay.md` §1.8, risk #1): on that client a "Disconnect"
    button sets a flag no authenticator reads, the live socket stays up, and
    the 90-day token keeps working.

    Soft-delete (`revoked_at`), not DELETE, so "which Mac was connected when,
    and when did I disconnect it?" stays answerable. Same retention
    reasoning as `ConnectorPendingAction`.
    """

    __tablename__ = "desktop_devices"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    # ON DELETE CASCADE: deleting the account must take the device rows with
    # it, exactly as mig 060 established for extension_devices.
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Self-reported by the app and shown on the approval page. Untrusted
    # display text: stored truncated, and the page renders it as text.
    device_name: Mapped[str] = mapped_column(String(200), nullable=False)
    app_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    os_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    # "direct" | "mas" — the distribution flavour (ARCHITECTURE.md §7).
    # Recorded because the two flavours have different capability sets (a MAS
    # build ships Files only), so an activity log that does not know which
    # build it is talking to cannot explain why a command was refused.
    flavor: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    # The `jti` of the currently-issued device token. The denylist is the
    # absence of a match: a token whose jti is not THIS value, or whose row
    # is revoked, does not authenticate. Storing the jti rather than the
    # token means a leaked database row is not a usable credential.
    token_jti: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True,
    )
    # When the currently-issued token dies. Kept as a column so the platform
    # can answer "this device needs to re-pair" without decoding a JWT, and
    # so a re-issue is one UPDATE.
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    paired_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    # Bumped by the tenant agent's heartbeat (connect / periodic / close),
    # which is the half of the extension's cross-process split that WAS
    # fixed (`agent-tool-relay.md` §1.9) and the reason presence is legible
    # from the platform at all. Read by the devices list AND by the approve
    # endpoint, which refuses to claim an action for an offline Mac.
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, index=True,
    )

    # `hw.model` as the Mac reported it at pairing ("Mac15,3"). Display only.
    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # "app" (a phone scanned the Mac's QR) or "web" (the typed code). Audit
    # only; nothing branches on it.
    paired_via: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    # The account session (JWT jti) that approved this Mac. CONNECTIONS.md S4.
    paired_session_jti: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    # The jti a re-issue just replaced, honoured until `prev_jti_expires_at`
    # so a socket authenticated with it survives the Mac's reconnect onto the
    # new token. Cleared by revoke together with `token_jti`: a revoked Mac
    # must not keep a second way in for two minutes.
    prev_token_jti: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True,
    )
    prev_jti_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    __table_args__ = (
        Index("ix_desktop_devices_user_revoked", "user_id", "revoked_at"),
    )


class DesktopPairing(Base):
    """One in-flight device-code pairing. Ten minutes, then it is dead.

    Persisted rather than held in memory for the replica reason in
    PAIRING_STATUSES above. Two more properties the protocol requires:

      * `device_code` is the bearer of the poll (§1.2) and is therefore
        indexed and unique, but it is NOT a credential for anything else —
        a successful poll marks the row `consumed` so the bundle cannot be
        replayed (the extension evicts its dict entry for the same reason,
        `extension.py:502-507`).
      * `approved_token`/`approved_device_id` hold the minted bundle between
        approve and the device's next poll. They are the one place a live
        token sits at rest in this schema; the row goes `consumed` on first
        read and the columns are cleared with it.
    """

    __tablename__ = "desktop_pairings"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    # >=128 bits of opaque, per §1.1.
    device_code: Mapped[str] = mapped_column(
        String(128), nullable=False, unique=True, index=True,
    )
    # Human-typeable, ambiguity-free, dashed — "WXYZ-1234".
    user_code: Mapped[str] = mapped_column(
        String(16), nullable=False, unique=True, index=True,
    )

    device_name: Mapped[str] = mapped_column(String(200), nullable=False)
    app_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    os_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    flavor: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True,
    )
    # Set at approve. Nullable because a pending row has no user yet — which
    # is also why `pair/init` is unauthenticated (§1.1: the device holds no
    # credential at that point).
    user_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )

    approved_device_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True,
    )
    approved_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    approved_expires_in: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    # Hard expiry. §1.1's `expires_in` is derived from this, and `poll`
    # applies it lazily on read for the same reason the connector list does
    # (`connector_pending_actions.py:407-412`): the moment someone is looking
    # at the record is the moment we know it is stale.
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    # Rate-limits the poll to the interval the init response advertised.
    # §1.2: a device that polls too fast gets `slow_down` and adds 5 s
    # permanently, per the OAuth device-flow convention.
    last_polled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    model: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    # The Mac's own account session. A QR challenge can never be claimed by
    # this session (CONNECTIONS.md S2): a script inside the Mac's web view
    # must not be able to start AND finish a pairing on its own.
    init_session_jti: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    # sha256 of the live QR challenge. The raw value is never stored, so a
    # leaked row cannot be replayed. One live challenge per pairing: minting
    # overwrites this and the previous QR stops working at that moment.
    qr_challenge_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True,
    )
    qr_minted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    qr_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    # Set by the first successful claim, which is what makes the challenge
    # single-use: every later claim from any other session is guarded out.
    qr_claimed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    qr_claimed_session_jti: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    # The phone's own label, flattened. Shown on the Mac, never trusted.
    qr_claimed_label: Mapped[Optional[str]] = mapped_column(
        String(60), nullable=True,
    )
    # Only the claiming session may decide, and only until this moment.
    claim_expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'denied', 'expired', "
            "'consumed')",
            name="ck_desktop_pairings_status",
        ),
        Index("ix_desktop_pairings_status_expires", "status", "expires_at"),
    )


class DesktopPendingAction(Base):
    """One staged, user-confirmable local action — a command, a write, a click.

    A SIBLING of `ConnectorPendingAction`, not a row in it, and not a
    `surface` column on it. The invariants are copied deliberately and
    item for item (`agent-tool-relay.md` §4.1): atomic claim guarded on
    `status = 'pending'`, hard `expires_at`, `tool_name` as a COLUMN rather
    than a payload field so the approve endpoint can never be talked into
    running a different tool than the card showed, and rows retained after
    they go terminal because they ARE the audit trail.

    WHY A SIBLING TABLE
    ───────────────────
    Two reasons, both read off the existing model rather than assumed:

    1. `ConnectorPendingAction.connector_id` is `nullable=False`, and the
       approve route around it resolves that id to a `ConnectorIdentity`,
       refreshes an OAuth token and calls `connector_dispatcher.execute`.
       A local action has no connector, no identity and no provider. Filling
       `connector_id` with a device id would put a value in a column whose
       stated job is to pin the executor, and point it at the wrong one.

    2. `GET /api/connectors/pending-actions` lists by user and status. Adding
       a discriminator column means every existing query has to learn to
       filter on it, and until they all do, the connector list serves desktop
       rows to a client that will POST them to the connector approve route.
       One route, two executors, and the wrong one picked by default is a
       worse failure than two tables.

    THE ONE STRUCTURAL DIFFERENCE, AND THE `dispatched` STATE
    ─────────────────────────────────────────────────────────
    `ConnectorPendingAction` executes ON THE PLATFORM after approval — its
    docstring names that as the reason it lives in the platform DB. A local
    action cannot: the executor is the user's Mac, reachable only over the
    relay socket, which is held by the tenant agent. So approval relays.

    `agent-tool-relay.md` §4.2 calls out the failure mode that creates
    ("approved-but-undeliverable") and gives two ways out. Both are taken,
    because neither alone is enough:

      * approval REQUIRES the device to be online at tap time, and refuses
        with a sentence otherwise. This is the safe default the audit
        recommends: "refuse the tap with 'your Mac is offline' rather than
        leave a command that might fire hours later when the laptop wakes."
      * and `dispatched` records that the relay accepted it, so a row left
        `approved` is legible as "claimed, never handed over" instead of
        being indistinguishable from "handed over, outcome unknown".

    And the device may still refuse. The server's approval is a
    PRECONDITION, never an authorisation: the Mac re-validates the payload
    against its own grants on every approved task and may answer `denied`.
    That is recommendation 10, and it is why `result_json` can record a
    refusal against a row this table marked approved.
    """

    __tablename__ = "desktop_pending_actions"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # The device the action was staged FOR. An approval must not be
    # deliverable to a Mac other than the one the agent was talking to when
    # it asked — pinned as a column for the same reason `tool_name` is.
    device_id: Mapped[str] = mapped_column(String(36), nullable=False)

    # Full namespaced name, e.g. "desktop__exec_run".
    tool_name: Mapped[str] = mapped_column(String(128), nullable=False)
    # JSON-encoded tool arguments and NOTHING else. Deliberately not a
    # free-form envelope: the approve request body carries no arguments at
    # all (see the route), so there is no editable-fields surface to get
    # wrong for a tool whose argument is a shell command.
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)

    # One plain sentence from the agent saying why. Untrusted display text —
    # §5.3: the device flattens control characters and caps it at 240 chars
    # before showing it, and so does the writer here.
    reason: Mapped[Optional[str]] = mapped_column(String(240), nullable=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="pending", index=True,
    )
    # The channel that staged it, recorded for the audit the way
    # `ConnectorPendingAction.channel` is.
    channel: Mapped[str] = mapped_column(
        String(32), nullable=False, default="desktop",
    )
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, index=True,
    )
    message_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # The relay task id, minted at stage time and reused verbatim when the
    # approved action is handed to the device. §5.3: `id` is the seam's
    # idempotency key and a replayed id is answered from the device's result
    # cache rather than executed twice — so a retried dispatch of an approved
    # row cannot run `rm` a second time. Minting a fresh id on retry would
    # defeat that completely; it is the same failure as regenerating a
    # `client_msg_id` on a chat replay.
    task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    decided_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    # 'web' | 'app' | 'mobile' — where the tap came from.
    decided_via: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    dispatched_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    # The device's own `status` + `summary`, never its `data`. The payload is
    # for the model; this column is the audit, and the audit records THAT a
    # tool ran — tool, target, decision, exit status, byte counts — and never
    # contents (§8, and ARCHITECTURE.md §5.8).
    result_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # The remote task this card belongs to, when a phone task staged it.
    # NULL for a card staged from ordinary chat.
    remote_task_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True, index=True,
    )
    decided_session_jti: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    # When the Mac said it is asking someone at the Mac. Stamped only while
    # the row is `dispatched`; it is what the phone reads as "At the Mac".
    mac_prompt_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'approved', 'dispatched', 'executed', "
            "'failed', 'rejected', 'expired')",
            name="ck_desktop_pending_actions_status",
        ),
        Index("ix_desktop_pending_actions_user_status", "user_id", "status"),
    )


# ─── Remote tasks (CONNECTIONS.md §8) ────────────────────────────────────
DESKTOP_TASK_STATUSES = (
    "queued", "dispatched", "running", "waiting_on_you", "waiting_on_mac",
    "done", "failed", "cancelled", "mac_offline",
)
DESKTOP_TASK_TERMINAL = frozenset({"done", "failed", "cancelled", "mac_offline"})
#: What the agent last said about the turn. `status` is DERIVED from this,
#: the task's cards and the Mac's presence by `desktop_task_state.derive`,
#: and never written from anything else.
DESKTOP_TASK_TURN_STATES = (
    "queued", "accepted", "running", "ended", "errored", "mac_offline",
)


class DesktopTask(Base):
    """One request typed on the phone for one Mac.

    PLATFORM_ONLY, beside `desktop_devices` and `desktop_pending_actions`,
    because its status is derived from both and the phone reads all three
    from the platform. It is not a `build_jobs` row: that table is
    AGENT_ONLY, and a task has to stay legible while its agent is down.

    Only a 120-character title is kept here. The full text goes to the agent
    and lives in the tenant conversation with every other chat, so this
    table never becomes a second copy of what the user asked their Mac.
    """

    __tablename__ = "desktop_tasks"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    device_id: Mapped[str] = mapped_column(String(36), nullable=False)
    title: Mapped[str] = mapped_column(String(120), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued", index=True,
    )
    turn_state: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued",
    )
    outcome: Mapped[Optional[str]] = mapped_column(String(240), nullable=True)
    conversation_id: Mapped[Optional[str]] = mapped_column(
        String(36), nullable=True,
    )
    # Client-minted idempotency key. A phone that retries a submit after a
    # dropped response gets the task it already created, never a second one.
    client_request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    submitted_session_jti: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True,
    )
    submitted_via: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    cancelled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True,
    )
    # "user" or "revoked": the two cancellations read differently.
    cancel_reason: Mapped[Optional[str]] = mapped_column(
        String(16), nullable=True,
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'dispatched', 'running', 'waiting_on_you', "
            "'waiting_on_mac', 'done', 'failed', 'cancelled', 'mac_offline')",
            name="ck_desktop_tasks_status",
        ),
        UniqueConstraint(
            "user_id", "client_request_id",
            name="uq_desktop_tasks_user_client_request",
        ),
        Index(
            "ix_desktop_tasks_user_device_created",
            "user_id", "device_id", "created_at",
        ),
    )
