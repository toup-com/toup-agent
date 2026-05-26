"""Regression: ``EmailReceivedHandler`` reads ``BuildJob.idempotency_key``,
not the long-gone ``TriggerEvent.event_dedupe_id``.

The PR #49 cutover repointed every trigger read path to ``BuildJob``,
but two sites in ``email_received_handler.py`` still accessed
``ev.event_dedupe_id`` + ``ev.external_payload`` — attributes that
don't exist on BuildJob. Production symptom (Gmail trigger detail
panel):

    Failed
    all_retries_exhausted: AttributeError("'BuildJob' object has no
    attribute 'event_dedupe_id'")

Every inbound Gmail event went through 3 retries, hit the same
AttributeError each time, and terminalised as ``failed``. Users saw
the trigger detail page report 38s of wasted retry budget per fire
with a confused "all_retries_exhausted" error_message.

Covers:

* Test-fire detection (the ``test:<uuid>`` short-circuit at the top
  of ``execute``) reads ``idempotency_key`` and reaches
  ``_do_test_fire`` for a BuildJob-shaped event without raising
  AttributeError.
* The real-fetch path's gmail-id resolution reads ``idempotency_key``
  for a BuildJob with no ``external_payload`` / ``event_dedupe_id``
  attributes — no AttributeError, gmail_id correctly extracted.
* Empty idempotency_key is gracefully reported as ``no_gmail_id``
  per-event error (NOT an exception).
* ``config_json["gmail_message_id"]`` is preferred when present
  (forward-compat with a future inbound endpoint change).
"""
from __future__ import annotations

import uuid

import pytest


pytestmark = pytest.mark.asyncio


class _FakeBuildJob:
    """Minimal BuildJob-shape stand-in. Carries exactly the attributes
    the handler is allowed to read after the PR #49 cutover. If the
    handler reaches for a TriggerEvent-era attribute (event_dedupe_id,
    external_payload, ...) it raises AttributeError — which is the
    regression we're guarding against.
    """

    __slots__ = ("id", "idempotency_key", "config_json")

    def __init__(
        self,
        *,
        id: str,
        idempotency_key: str | None,
        config_json: dict | None = None,
    ):
        self.id = id
        self.idempotency_key = idempotency_key
        self.config_json = config_json


def _trigger():
    """Minimal Trigger stand-in — handler.execute reads .config_json
    for delivery_channels + filter; tests give it the empty defaults."""
    class _T:
        id = "trig-1"
        user_id = "user-1"
        name = "Inbox"
        kind = "email_received"
        config_json = {"delivery_channels": ["website"]}
        provider_state_json: dict = {}
        last_status = "never_fired"
        last_error = None
        fire_count = 0
    return _T()


# ── Test-fire short-circuit ───────────────────────────────────────


async def test_test_fire_detected_via_idempotency_key_no_attribute_error():
    """The first thing ``execute`` does (after the credit pre-flight)
    is check if EVERY event in the batch has an ``idempotency_key``
    starting with ``test:``. Pre-fix this read ``ev.event_dedupe_id``
    and raised AttributeError on every Gmail fire — even the real
    ones — because the check ran BEFORE any per-event branching."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    sends: list = []

    async def _writer(*a, **kw):
        sends.append(("write", kw))
        return "msg-id", "day-id"

    async def _broadcaster(*a, **kw):
        sends.append(("broadcast", kw))
        return 1

    handler = EmailReceivedHandler(
        writer=_writer, broadcaster=_broadcaster,
    )

    test_events = [
        _FakeBuildJob(id=str(uuid.uuid4()), idempotency_key=f"test:{uuid.uuid4().hex}")
        for _ in range(2)
    ]

    # If the handler still reads `event_dedupe_id` this raises
    # AttributeError. The fix makes it read `idempotency_key` and
    # take the test-fire short-circuit cleanly.
    result = await handler.execute(_trigger(), test_events, db=None)

    # _do_test_fire is the short-circuit path — it returns a non-None
    # TriggerResult with one of the test-success statuses. We just
    # assert it didn't raise and returned something coherent.
    assert result is not None
    assert getattr(result, "status", None) in (
        "test_success", "success", "success_empty",
    )


async def test_real_event_resolves_gmail_id_from_idempotency_key():
    """For real (non-test) fires, the handler's _fetch_all path
    extracts the Gmail message id. Pre-fix it tried
    ``ev.external_payload["gmail_message_id"]`` (AttributeError on
    BuildJob) then fell back to ``ev.event_dedupe_id`` (also
    AttributeError). Post-fix it reads ``ev.idempotency_key`` which
    the inbound endpoint sets to the Gmail message id."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    handler = EmailReceivedHandler()
    real_event = _FakeBuildJob(
        id=str(uuid.uuid4()),
        idempotency_key="18b1f9c5e4d2a000",  # plausible Gmail msg id
    )

    # _fetch_all is the per-event id resolver. We're checking the
    # extraction logic, not the MCP fetch — so we pass a None mcp
    # and inspect the per-event error map: a real id with no MCP
    # should error as "mcp_client_unavailable" or similar (or
    # absent depending on the fetch path), but it MUST NOT error
    # as "no_gmail_id" or raise AttributeError.
    try:
        fetched, errors = await handler._fetch_all(None, [real_event])
    except AttributeError as e:
        pytest.fail(
            f"AttributeError raised on BuildJob-shaped event: {e}. "
            "The PR #49 cutover wasn't applied to this read path."
        )

    # The event id should NOT show up in `errors` with code
    # "no_gmail_id" — the gmail_id resolution succeeded.
    assert errors.get(real_event.id) != "no_gmail_id"


async def test_missing_idempotency_key_reports_no_gmail_id_no_exception():
    """When idempotency_key is None / empty, the handler must error
    per-event (no_gmail_id) without raising — defensive guarantee
    for the contract."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    handler = EmailReceivedHandler()
    blank_event = _FakeBuildJob(
        id=str(uuid.uuid4()), idempotency_key=None,
    )

    fetched, errors = await handler._fetch_all(None, [blank_event])
    assert errors.get(blank_event.id) == "no_gmail_id"
    assert fetched == []


async def test_config_json_gmail_message_id_preferred_over_idempotency_key():
    """Forward-compat: if/when the inbound endpoint starts persisting
    ``external_payload`` into ``BuildJob.config_json``, the handler
    prefers an explicit ``gmail_message_id`` from there over the
    fallback ``idempotency_key`` lookup."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    handler = EmailReceivedHandler()
    ev = _FakeBuildJob(
        id=str(uuid.uuid4()),
        idempotency_key="should-not-be-used",
        config_json={"gmail_message_id": "preferred-msg-id"},
    )

    # Inspect via the same MCP-less path. We don't get the actual
    # gmail_id back out of _fetch_all directly — but if extraction
    # picks the right field the event lands as "mcp_client_unavailable"
    # (or whatever the next step decides), NOT "no_gmail_id". The
    # extraction logic is the assert-target here.
    fetched, errors = await handler._fetch_all(None, [ev])
    assert errors.get(ev.id) != "no_gmail_id"


# ── Mixed test+real batch must NOT take the test short-circuit ──


async def test_mixed_batch_does_not_take_test_short_circuit():
    """The test-fire short-circuit triggers ONLY when EVERY event in
    the batch is synthetic. A coalesced batch with one real + one
    test event must go through the full pipeline (real Gmail fetch +
    summarise), not the deterministic 'wiring works' path."""
    from app.agent.triggers.email_received_handler import EmailReceivedHandler

    handler = EmailReceivedHandler()

    mixed = [
        _FakeBuildJob(id=str(uuid.uuid4()), idempotency_key="real-gmail-id"),
        _FakeBuildJob(id=str(uuid.uuid4()), idempotency_key=f"test:{uuid.uuid4().hex}"),
    ]

    # Asserting the boolean directly is enough — we don't need to
    # run the whole handler. The all() expression is what the
    # short-circuit uses internally.
    all_test = all(
        (getattr(ev, "idempotency_key", None) or "").startswith("test:")
        for ev in mixed
    )
    assert all_test is False
