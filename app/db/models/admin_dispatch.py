"""Admin dispatch — operator→user announcements (platform-only).

Three tables, all PLATFORM_ONLY (see base.py):

- ``admin_dispatches``: one row per compose-and-send from the admin panel.
  ``audience`` is one user or everyone; ``mode`` decides whether the in-chat
  card is retracted once read (``once``) or stays, carrying a Reply action
  (``persistent``).
- ``admin_dispatch_targets``: the per-recipient fan-out ledger, one row per
  user the dispatch is owed to. Rows are CLAIMED with a state CAS
  (``UPDATE … WHERE state='pending'``) because platform-api runs 2 Railway
  replicas with no leader election — the same stance ``notification_queue``
  takes (notification.py:17-20).
- ``admin_thread_messages``: the ongoing operator↔user thread a
  ``persistent`` dispatch opens, both directions.

Why the thread lives on the platform and not in the tenant DB: it is a
conversation between the OPERATOR and the user, not agent data. The agent
must never read it (the in-chat row it *does* write carries
``day_chat_id = NULL``, so it is structurally invisible to
``load_day_context``), the admin panel must be able to read replies across
all users without fanning out to N containers, and a reply has to keep
working while the user's agent container is down.

Read receipts live on ``admin_dispatch_targets.read_at`` and nowhere else.
The tenant ``Message`` row carries no read state, so there is exactly one
place to look — and ``once`` retraction keys off it.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, DateTime, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# ── Dispatch vocabulary ───────────────────────────────────────────
# `once` is a real lifecycle, not a render rule: on read the platform
# proxies a retract to the agent, which DELETEs the row and broadcasts
# `admin_notice_retract`. `persistent` rows stay and open a thread.
DISPATCH_MODE_ONCE = "once"
DISPATCH_MODE_PERSISTENT = "persistent"
DISPATCH_MODES: set[str] = {DISPATCH_MODE_ONCE, DISPATCH_MODE_PERSISTENT}

DISPATCH_AUDIENCE_USER = "user"   # target_user_id set
DISPATCH_AUDIENCE_ALL = "all"     # every user; unrecallable, hence the confirm gate
DISPATCH_AUDIENCES: set[str] = {DISPATCH_AUDIENCE_USER, DISPATCH_AUDIENCE_ALL}

DISPATCH_QUEUED = "queued"    # accepted, fan-out not started
DISPATCH_SENDING = "sending"  # worker walking the targets
DISPATCH_SENT = "sent"        # every target terminal
DISPATCH_FAILED = "failed"    # fan-out itself died; per-target detail in the ledger
DISPATCH_STATUSES: set[str] = {
    DISPATCH_QUEUED, DISPATCH_SENDING, DISPATCH_SENT, DISPATCH_FAILED,
}

# ── Per-target state ──────────────────────────────────────────────
TARGET_PENDING = "pending"
TARGET_SENDING = "sending"    # CAS-claimed by one replica
TARGET_DONE = "done"
TARGET_FAILED = "failed"
TARGET_STATES: set[str] = {
    TARGET_PENDING, TARGET_SENDING, TARGET_DONE, TARGET_FAILED,
}

# The chat surface tracks its own outcome: the notification lane can land
# while the agent hop does not, and `no_agent` is a recorded fact rather
# than a failure — a user with no running container still gets the push.
CHAT_PENDING = "pending"
CHAT_DELIVERED = "delivered"
CHAT_NO_AGENT = "no_agent"
CHAT_FAILED = "failed"
CHAT_RETRACTED = "retracted"
CHAT_STATUSES: set[str] = {
    CHAT_PENDING, CHAT_DELIVERED, CHAT_NO_AGENT, CHAT_FAILED, CHAT_RETRACTED,
}

# ── Thread direction ──────────────────────────────────────────────
THREAD_OUT = "out"  # admin → user
THREAD_IN = "in"    # user → admin
THREAD_DIRECTIONS: set[str] = {THREAD_OUT, THREAD_IN}


# ── Message origin — WHO the message is from, as a party ──────────
# Not the surface it arrived on (`Message.channel`) and not the feature that
# produced it (`Message.source`). Three parties can push a message the user
# did not ask for, and they differ in exactly one way that matters:
#
#   admin   a human operator. NEVER enters agent context — an operator's
#           words reaching the model is prompt injection with a human author.
#   agent   the user's own agent, proactively. BELONGS in context: it is the
#           agent's own prior utterance, and hiding it makes the agent
#           contradict itself. Reserved; nothing writes it yet.
#   system  the platform speaking about itself (billing, incidents).
#
# The exclusion in `day_context_loader` filters on ORIGIN_ADMIN specifically
# and never on "did the dispatch path write this". That distinction is the
# whole reason this vocabulary exists: agent-initiated contact is meant to
# reuse the same write → fan-out → notify → ack → revoke path by writing a
# different value here and changing nothing else.
ORIGIN_ADMIN = "admin"
ORIGIN_AGENT = "agent"
ORIGIN_SYSTEM = "system"
MESSAGE_ORIGINS: set[str] = {ORIGIN_ADMIN, ORIGIN_AGENT, ORIGIN_SYSTEM}

# The origins whose rows must never reach an LLM. A set, not a bare
# `!= 'admin'`, so adding a future non-model-visible party is one edit here
# rather than a grep for every comparison.
ORIGINS_EXCLUDED_FROM_CONTEXT: frozenset[str] = frozenset({ORIGIN_ADMIN})


def _default_sender_name() -> str:
    """Deferred settings read — see base.py::_get_embedding_dim for why.

    Reading `settings` at class-body time makes the model module depend on
    config import order; at flush time it is always available.
    """
    from app.config import settings
    return settings.admin_dispatch_sender_name


class AdminDispatch(Base):
    """One operator send.

    The id is generated by the caller (`str(uuid.uuid4())`) before the row
    is flushed, not by the column default: the notification
    `idempotency_key`, the Live Activity `mission_id` and the tenant
    message's uuid5 are all derived from it, so it has to be known while
    the request is still assembling its side effects.

    The four counters are the admin panel's whole progress view. They are
    maintained by the fan-out worker as each target goes terminal, which
    means they can trail reality by one target — a broadcast is a long
    walk, and a snapshot that is briefly low is better than holding a
    transaction open across N agent hops.
    """

    __tablename__ = "admin_dispatches"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    # ON DELETE SET NULL: an audit reference to the admin who sent it.
    # Deleting that account must not erase the dispatch or its counters.
    created_by_user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
    )
    mode: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DISPATCH_MODE_ONCE,
    )
    # The party this send is from. Every row written today is 'admin'; the
    # column exists so the delivery path (write → fan-out → notify → ack →
    # revoke) is origin-AGNOSTIC before anything else needs it. A future
    # agent-initiated message is meant to be one row with a different value
    # here and no new code — which is only true if identity, badge and copy
    # are already driven by this field rather than hardcoded to the operator.
    origin: Mapped[str] = mapped_column(
        String(16), nullable=False, default=ORIGIN_ADMIN, index=True,
    )
    audience: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DISPATCH_AUDIENCE_USER,
    )
    # Set iff audience == 'user'. CASCADE: a single-recipient dispatch has
    # nothing left to say once that recipient is gone.
    target_user_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True,
    )
    # Shown on the card in place of the agent's name — the message is from
    # the operator, and the whole feature rests on the user seeing that.
    sender_name: Mapped[str] = mapped_column(
        String(80), nullable=False, default=_default_sender_name,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    # Bypasses quiet hours and the daily cap. Operator judgement, recorded
    # per dispatch rather than inferred from the copy.
    urgent: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # A catalogue id from `app/services/dispatch_tones.py`, never a
    # filename — the file a given id maps to is a property of the app
    # build, and a row that stored `toup_ping.caf` would outlive the
    # release that bundled it. NULL is the pre-088 history and every
    # reader collapses it to the system default.
    tone: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DISPATCH_QUEUED, index=True,
    )
    target_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # "≥1 surface landed" — chat or notification, not both.
    delivered_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    read_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Why THIS ROW reads `failed` — a fan-out-level fact, never a per-target
    # one (those live on `admin_dispatch_targets.last_error`). The stall
    # sweeper is the reason it has to be persisted at all: a worker that was
    # OOM-killed or redeployed runs none of its own error handling, so there
    # are no target rows and no exception text, and "Failed" with no reason is
    # what the operator was left staring at. Cleared at the top of every
    # fan-out so a retry cannot inherit the previous attempt's reason.
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # ── Recall (migration 091) ───────────────────────────────────────
    # A recall is an ACT BY A PERSON, and the question afterwards is always
    # "who, and when" — which a boolean answers neither of.
    #
    # It removes the CARD from every recipient's chat. It cannot un-send the
    # NOTIFICATION: nothing takes a delivered push off a lock screen. The two
    # halves of "sent" are recallable to different degrees and the compose copy
    # now says so instead of flattening both into "cannot be recalled".
    #
    # `revoked_by_user_id` is a plain String, not a CASCADE foreign key, for
    # the same reason `created_by_user_id` is: an operator's account being
    # deleted must not erase the record that they recalled a broadcast.
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    revoked_by_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)


class AdminDispatchTarget(Base):
    """One recipient of one dispatch — the fan-out ledger row.

    Claimed with a state CAS, never ``FOR UPDATE SKIP LOCKED``: two
    replicas race on the same row and exactly one wins, and the sqlite
    test harness has no row-locking to model.

    ``chat_status`` is tracked separately from ``state`` because the two
    surfaces fail independently: a user whose agent container is down
    still gets the notification, and that is a ``done`` target with
    ``chat_status='no_agent'`` — a recorded fact, not a failure.
    """

    __tablename__ = "admin_dispatch_targets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    dispatch_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("admin_dispatches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    state: Mapped[str] = mapped_column(
        String(16), nullable=False, default=TARGET_PENDING, index=True,
    )
    chat_status: Mapped[str] = mapped_column(
        String(24), nullable=False, default=CHAT_PENDING,
    )
    # The tenant Message id — deterministic (uuid5 of dispatch+user), so a
    # replayed hop collides on the PK instead of writing a second card.
    chat_message_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    notification_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # The only read receipt in the system (see module docstring).
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow,
    )

    __table_args__ = (
        # Replay guard for the retry sweep: re-running a dispatch must
        # never add a second row for a user it already reached.
        UniqueConstraint(
            "dispatch_id", "user_id", name="uq_admin_dispatch_target",
        ),
        # The worker's claim scan: pending rows of one dispatch.
        Index("ix_admin_dispatch_targets_claim", "state", "dispatch_id"),
    )


class AdminThreadMessage(Base):
    """One message in the operator↔user thread.

    ``dispatch_id`` deliberately carries NO foreign key: it is a
    provenance stamp ("this row is the one that persistent dispatch
    opened"), and the user's thread must survive the dispatch row being
    pruned. A user reply has none at all.

    Two read stamps because the thread has two readers with unrelated
    unread counts: ``read_at`` is the user reading an ``out`` row (drives
    the rail badge), ``admin_read_at`` is the operator reading an ``in``
    row (drives the panel's reply queue).
    """
    # NOTE: the 085 migration must carry every column here verbatim. A name
    # that differs between the two is invisible until a platform DB built by
    # `create_all` meets one built by alembic.

    __tablename__ = "admin_thread_messages"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dispatch_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    direction: Mapped[str] = mapped_column(String(4), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    author_admin_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    # The operator's display name AS IT WAS when the row was written — one
    # operator identity across both surfaces. The chat card renders the
    # dispatch's `sender_name`; with nothing here the thread had to invent one
    # (mobile hardcoded "Toup", web showed none), so the same operator reached
    # the user as two different parties. Copied from the dispatch on the row a
    # persistent dispatch opens, and from settings on an admin follow-up.
    #
    # Nullable because an `in` row is the USER's own words and has no operator
    # name; serialisation also falls back to the configured default for an
    # `out` row that somehow has none, since rendering the operator nameless is
    # the very defect this column closes.
    sender_name: Mapped[Optional[str]] = mapped_column(String(80), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, index=True,
    )
    read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    admin_read_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # ── Deletion (migration 092) ─────────────────────────────────────
    #
    # TWO stamps, because "delete" means two different things and an operator
    # has to be able to say which:
    #
    #   hidden_from_user_at — gone for THEM, still in the operator's thread,
    #     marked. This is the one an operator almost always wants: the message
    #     stops being visible to the person it was sent to, and the record of
    #     having sent it survives.
    #   deleted_at — gone for BOTH. The body is cleared, and a tombstone row
    #     remains so the thread does not silently lose a turn.
    #
    # NEITHER is a hard DELETE. A thread is a conversation between two people
    # and one of them cannot un-remember it; removing the row outright would
    # also renumber nothing but silently drop the operator's own audit trail,
    # which is the thing most needed when a message was sent by mistake.
    #
    # `deleted_at` set implies the user cannot see it either — the read filter
    # is `hidden_from_user_at IS NULL AND deleted_at IS NULL`, so `both` does
    # not need to set the first one as well.
    hidden_from_user_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deleted_by_user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)

    __table_args__ = (
        # Both readers page the same way: one user's thread, newest last.
        Index("ix_admin_thread_user_created", "user_id", "created_at"),
    )
