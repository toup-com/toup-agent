"""Pin the wire-format of datetime fields in trigger + routine responses.

The 2026-05-14 dashboard bug: events stamped via ``datetime.utcnow()``
(naive UTC) were serialised as ``"2026-05-14T19:18:00"`` with no
timezone marker. ``new Date(s)`` in the browser parses that as LOCAL
time → events fired at 19:18 UTC rendered as "7:18 PM" on a screen
where the wall clock read 3:18 PM EDT.

Fix: ``app.api.{triggers,routines}._utc_iso`` forces every datetime
emitted to the frontend to end with ``Z``. These tests pin that
behaviour so a refactor can't quietly drop the serialisers.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


# ── helper unit tests ───────────────────────────────────────────────


def test_utc_iso_handles_naive_datetime():
    """A naive UTC datetime (``datetime.utcnow()``) is the common case
    on the trigger / routine writers. It must serialise with a Z."""
    from app.api.triggers import _utc_iso as triggers_iso
    from app.api.routines import _utc_iso as routines_iso

    naive = datetime(2026, 5, 14, 19, 18, 0)
    assert triggers_iso(naive) == "2026-05-14T19:18:00Z"
    assert routines_iso(naive) == "2026-05-14T19:18:00Z"


def test_utc_iso_handles_aware_datetime_in_utc():
    """A tz-aware datetime already at UTC must still emit Z, not
    ``+00:00`` — keep the wire format consistent so the frontend
    parser only has to handle one shape."""
    from app.api.triggers import _utc_iso
    aware_utc = datetime(2026, 5, 14, 19, 18, 0, tzinfo=timezone.utc)
    out = _utc_iso(aware_utc)
    assert out.endswith("Z"), out
    assert "+00:00" not in out


def test_utc_iso_converts_non_utc_aware_to_utc():
    """A datetime in another timezone must be converted to UTC before
    rendering. Otherwise the wire field claims UTC but carries local
    wall-clock time."""
    from datetime import timedelta
    from app.api.triggers import _utc_iso
    edt = timezone(timedelta(hours=-4))
    aware_edt = datetime(2026, 5, 14, 15, 18, 0, tzinfo=edt)
    # 15:18 EDT = 19:18 UTC
    assert _utc_iso(aware_edt) == "2026-05-14T19:18:00Z"


def test_utc_iso_handles_none():
    """Optional[datetime] fields pass None through to None — Pydantic
    won't render the JSON key at all."""
    from app.api.triggers import _utc_iso as triggers_iso
    from app.api.routines import _utc_iso as routines_iso
    assert triggers_iso(None) is None
    assert routines_iso(None) is None


# ── End-to-end Pydantic serialisation ───────────────────────────────


def test_trigger_event_response_emits_z_suffix():
    """The actual response model the API returns must emit Z on every
    datetime field. Without this the frontend renders the wrong time."""
    from app.api.triggers import TriggerEventResponse

    naive = datetime(2026, 5, 14, 19, 18, 0)
    resp = TriggerEventResponse(
        id="ev-1",
        trigger_id="t-1",
        event_dedupe_id="test:abc",
        received_at=naive,
        started_at=naive,
        finished_at=naive,
        status="success",
    )
    payload = resp.model_dump_json()
    # All three datetime fields must end with Z in the JSON output.
    assert payload.count("2026-05-14T19:18:00Z") == 3, payload


def test_trigger_response_emits_z_suffix():
    from app.api.triggers import TriggerResponse

    naive = datetime(2026, 5, 14, 19, 18, 0)
    resp = TriggerResponse(
        id="t-1",
        kind="email_received",
        action="summarize_and_post",
        name="Gmail summary",
        enabled=True,
        filter_json=None,
        config_json=None,
        provider_state_json=None,
        last_fired_at=naive,
        fire_count=1,
        last_status="active",
        last_error=None,
        created_at=naive,
        updated_at=naive,
        delivery_channels=["website"],
        watch_provisioned=True,
    )
    payload = resp.model_dump_json()
    # last_fired_at + created_at + updated_at = 3 datetime fields
    assert payload.count("2026-05-14T19:18:00Z") == 3, payload


def test_routine_response_emits_z_suffix():
    from app.api.routines import RoutineResponse

    naive = datetime(2026, 5, 14, 19, 18, 0)
    resp = RoutineResponse(
        id="r-1",
        kind="reminder",
        enabled=True,
        schedule_kind="cron",
        schedule_cron_local="0 9 * * *",
        config=None,
        last_state=None,
        last_run_at=naive,
        next_run_at=naive,
        last_status="never_run",
        last_error=None,
        created_at=naive,
        updated_at=naive,
    )
    payload = resp.model_dump_json()
    # last_run_at + next_run_at + created_at + updated_at = 4 fields
    assert payload.count("2026-05-14T19:18:00Z") == 4, payload
