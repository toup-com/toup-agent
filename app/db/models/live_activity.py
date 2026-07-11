"""iOS Live Activity models (Autopilot phone surface — platform-only).

Two tables, both PLATFORM_ONLY (see base.py):

- ``live_activity_devices``: one row per iOS device that registered an
  ActivityKit *push-to-start* token (iOS 17.2+, one token per app per
  device, obtained via ``Activity.pushToStartTokenUpdates``). The
  platform uses it to START a mission Live Activity via APNs even when
  the app is force-quit. Same isolation stance as ``push_devices``:
  APNs tokens live ONLY on the platform, never in agent containers.

- ``live_activities``: one row per (device, mission) Live Activity the
  platform has started. Updates are pushed to ``activity_push_token``
  when the app has reported one, else to the device's push-to-start
  token — the start payload carries ``"input-push-token": 1`` (iOS 18+)
  which makes that token double as the update token, so progress keeps
  flowing without the app ever waking. ``mission_id`` is also the
  ActivityAttributes ``name`` in the pushed payload, which is how the
  app's ``onTokenReceived`` listener maps a reported per-activity token
  back to this row.

Token environments: a dev-signed build (``npx expo run:ios``) gets an
``aps-environment=development`` entitlement and its tokens are only
valid against ``api.sandbox.push.apple.com``; TestFlight/App Store
builds are ``production``. The client reports its environment at
registration and every APNs send picks the host from the row —
mixing them up yields ``BadDeviceToken``.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    DateTime, ForeignKey, Index, Integer, String, UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# ── Live Activity statuses ────────────────────────────────────────
LA_STARTED = "started"   # start push accepted by APNs
LA_ENDED = "ended"       # end push sent (mission reached a terminal state)
LA_FAILED = "failed"     # APNs rejected the start (bad token / env mismatch)

# ── APNs environments ─────────────────────────────────────────────
APNS_ENV_DEVELOPMENT = "development"
APNS_ENV_PRODUCTION = "production"
KNOWN_APNS_ENVIRONMENTS: set[str] = {APNS_ENV_DEVELOPMENT, APNS_ENV_PRODUCTION}


class LiveActivityDevice(Base):
    """An iOS device registered for ActivityKit push-to-start.

    Upsert key is ``push_to_start_token``: Apple rotates it at its own
    discretion, so the app re-registers on every launch and each
    rotation lands as a fresh row while the stale one is revoked by
    APNs errors (410 / ``BadDeviceToken``) on next use. Re-registration
    of a known token bumps ``last_seen_at``, re-activates a revoked
    row, and re-binds to the signing-in user (last sign-in wins, same
    policy as ``push_devices``).
    """

    __tablename__ = "live_activity_devices"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Hex-encoded APNs token — 64 chars today, 200 leaves headroom.
    push_to_start_token: Mapped[str] = mapped_column(
        String(200), nullable=False, unique=True,
    )
    apns_environment: Mapped[str] = mapped_column(
        String(12), nullable=False, default=APNS_ENV_DEVELOPMENT,
    )
    device_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    app_version: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, index=True,
    )

    __table_args__ = (
        Index("ix_live_activity_devices_user_revoked", "user_id", "revoked_at"),
    )


class LiveActivity(Base):
    """A mission Live Activity the platform started on one device.

    ``last_progress`` (0-100) is kept so out-of-order dispatcher ticks
    never move the on-screen progress bar backwards.
    """

    __tablename__ = "live_activities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    # Agent-side Routine.id — opaque here (agent tables are unreadable
    # from the platform); also the ActivityAttributes ``name``.
    mission_id: Mapped[str] = mapped_column(String(64), nullable=False)
    device_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("live_activity_devices.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Per-activity update token, reported by the app when it observes
    # one (best-effort — see module docstring for the iOS 18 fallback).
    activity_push_token: Mapped[Optional[str]] = mapped_column(
        String(200), nullable=True,
    )
    apns_environment: Mapped[str] = mapped_column(
        String(12), nullable=False, default=APNS_ENV_DEVELOPMENT,
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=LA_STARTED, index=True,
    )
    last_progress: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False,
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "device_id", "mission_id", name="uq_live_activities_device_mission"
        ),
        Index("ix_live_activities_mission_status", "mission_id", "status"),
    )
