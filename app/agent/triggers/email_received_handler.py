"""Email-received trigger handler — kind #1.

Reacts to "new email arrived" Pub/Sub pushes. For each event in the
runner-claimed batch (a single message OR a coalesced burst), the
handler:

  1. Fetches the full Gmail message via the existing MCP
     `gmail__get_message` tool. Tokens stay platform-side; the agent's
     MCP client routes through the platform dispatcher with X-Agent-Key
     auth, so we get the same isolation as the email_briefing handler.
  2. Applies the user's filter rules (`filter_evaluator`). Drop on
     mismatch.
  3. Dispatches by `Trigger.action`:
        - **summarize_and_post**: LLM summary → Day-as-Chat.
        - **notify_only**: sender + subject + body snippet → Day-as-Chat.
        - **forward_to_telegram**: Day-as-Chat post + Telegram DM via
          the routines channel_dispatcher.

Coalescing: when N events arrive in one handler call (the runner
folded a sibling burst), `summarize_and_post` produces ONE digest
("You got N new emails: …"). `notify_only` produces one card per
email. `forward_to_telegram` digests like summarize.

Failure model: per-event in `TriggerResult.per_event_status`. Fetch
failure on event K doesn't sink K-1; it gets marked `failed` and the
remaining events continue. Reauth required short-circuits the whole
batch (every event is the same Gmail account; if auth is broken
once, it's broken for the rest).

Every batch result also carries `last_handler_completed_at` +
`last_handler_status` in `new_provider_state` so the runner stamps them
on the parent trigger row. The `/probe` diagnosis endpoint reads
those fields to tell "handler reached terminal state recently" from
"dispatched but no handler response (handler hang / runner crash)".
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .base_handler import TriggerResult
from .filter_evaluator import matches_filter, _flatten_headers
from .message_writer import write_trigger_message, broadcast_trigger_message


logger = logging.getLogger(__name__)


# Conservative cap — multi-email digests can run longer than single-email
# summaries. 1500 tokens ≈ 1100 words; Markdown formatted output for the
# Day-as-Chat surface.
_SUMMARY_MAX_TOKENS = 1500
_OPERATION_TYPE = "user.trigger.email_received"


# Per-process dedupe for reauth notices. Key: (user_id, trigger_id);
# value: epoch seconds of last notify. Pub/Sub retries + sibling pushes
# would otherwise spam the user with reconnect cards on every failed
# fetch. 24h TTL — long enough that a real retry storm doesn't repeat,
# short enough that the user gets a fresh reminder the next day if they
# haven't acted. In-memory is sufficient: a process restart issues at
# most one duplicate, and the typical fix is "reconnect Gmail" which
# stops the failures (and thus the notices) at the source.
_REAUTH_NOTICE_DEDUPE: dict[tuple[str, str], float] = {}
_REAUTH_NOTICE_TTL_SEC = 24 * 60 * 60

_REAUTH_NOTICE_CARD = (
    "**Heads up — Gmail trigger can't fire.**\n\n"
    "Your Gmail connection stopped working, so the trigger "
    "**{trigger_name}** is paused until you reconnect.\n\n"
    "Fix: open Toup → Settings → Integrations → Gmail → **Connect**.\n\n"
    "_(You'll get this notice at most once per day until it's fixed.)_"
)


_SUMMARIZE_SYSTEM_PROMPT = """You write the SUMMARY body of a Gmail notification card. The header (📧 sender + subject) is rendered by the surface; your job is only the bullet-content underneath.

Rules:
- Tight 1–3 sentence summary of what the email actually says.
- If body has ANY content (even one short line, even with typos, even with unfamiliar acronyms / names / jargon), produce a summary by paraphrasing or quoting the text directly. NEVER bail with "(no body)" when the body has content — the user wrote it and wants to see what they wrote reflected back.
- Only respond "(no body)" when the body block below is literally empty / whitespace-only.
- ONLY include an "Action: …" line when the email explicitly asks for a reply, deadline, approval, or RSVP.
- Multiple emails: list each as "• <one-line gist>" (sender comes from the header above, don't repeat it).
- ≤120 words for one email, ≤250 words for a burst. Hard cap 400.
- Markdown OK for emphasis (**bold** sparingly, ⚑ for boss/family/financial/security).
- NEVER write "From:", "Subject:", "(unknown)", "Here is a summary", or any opening boilerplate — the header card already covers that.

Never invent details. When unsure, quote the sender's exact words rather than paraphrasing or refusing."""


class _ReauthRequired(Exception):
    """Connector returned `kind=reauth_required` — short-circuit the
    batch."""


class _ToolMissing(Exception):
    """Gmail tool wasn't in tools/list. Treated as reauth."""


@dataclass
class _FetchedEmail:
    event_id: str           # the TriggerEvent.id that triggered the fetch
    gmail_id: str
    headers: dict
    snippet: str
    body: str
    labels: list[str]
    raw_message: dict       # full payload for filter eval


def _synthetic_test_email(event_id: str, gmail_id: str) -> _FetchedEmail:
    """Build an in-memory `_FetchedEmail` for a test-fire event.

    Test events have ``idempotency_key = "test:<uuid>"`` and aren't
    real Gmail messages — fetching them via MCP returns 404. Instead
    we synthesise a realistic email envelope so the rest of the
    pipeline (filter eval → summarise/notify/forward → Day-as-Chat
    write) runs end-to-end against a known-good payload.

    Result: clicking "Test" actually produces a chat message
    ("Trigger test — wiring works"), `last_status` flips to `active`,
    and the user gets a real signal that the trigger is alive instead
    of a cryptic Failed row.
    """
    subject = "[Test] Trigger wiring check"
    sender = "Toup Triggers <triggers@toup.ai>"
    body = (
        "This is a synthetic test event generated by the Trigger "
        '"Test" button on Mission Control.\n\n'
        "No real email was fetched. The handler ran end-to-end against "
        "this synthetic message so you can confirm:\n"
        "  • The trigger is registered with the runner\n"
        "  • The summariser / action pipeline executes\n"
        "  • Day-as-Chat delivery (and any Telegram / WhatsApp fan-out) works\n\n"
        "A real Gmail push will deliver the actual message — this test "
        "just proves the plumbing is alive."
    )
    headers = {
        "subject": subject,
        "from": sender,
        "to": "you@example.com",
        "date": "",
    }
    raw_message = {
        "id": gmail_id,
        "snippet": body[:200],
        "labelIds": ["INBOX", "UNREAD"],
        "payload": {
            "headers": [
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
                {"name": "To", "value": "you@example.com"},
            ],
            "body": {"data": "", "size": len(body)},
            "mimeType": "text/plain",
        },
    }
    return _FetchedEmail(
        event_id=event_id,
        gmail_id=gmail_id,
        headers=headers,
        snippet=body[:200],
        body=body,
        labels=["INBOX", "UNREAD"],
        raw_message=raw_message,
    )


def _unpack_ok(call_result: Any, tool_name: str) -> dict:
    """Lift the structured envelope out of an MCP call result.

    Same shape as `email_briefing_handler._unpack_ok`. Duplicated
    rather than imported to keep the trigger module's blast radius
    independent — a refactor of the routine helper must not
    accidentally change trigger behaviour."""
    envelope = getattr(call_result, "structured_content", None)
    if envelope is None:
        envelope = getattr(call_result, "structuredContent", None)
    if not isinstance(envelope, dict):
        raise RuntimeError(
            f"{tool_name}: missing structured envelope "
            f"(got {type(envelope).__name__})"
        )
    kind = envelope.get("kind")
    if kind == "ok":
        raw = envelope.get("content")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"{tool_name}: content not JSON ({e})"
                ) from e
        if isinstance(raw, dict):
            return raw
        raise RuntimeError(
            f"{tool_name}: unexpected content type {type(raw).__name__}"
        )
    if kind == "reauth_required":
        raise _ReauthRequired(envelope.get("message") or "reauth required")
    msg = envelope.get("message") or f"connector returned kind={kind}"
    raise RuntimeError(f"{tool_name}: {msg}")


class EmailReceivedHandler:
    """TriggerHandler for `kind="email_received"`."""

    kind = "email_received"

    def __init__(
        self,
        mcp_client: Any = None,
        llm_fn: Any = None,
        writer: Any = None,
        broadcaster: Any = None,
    ):
        self._mcp_client = mcp_client
        self._llm_fn = llm_fn
        self._writer = writer
        self._broadcaster = broadcaster

    async def execute(
        self,
        trigger: Any,
        events: list[Any],
        db: AsyncSession,
    ) -> TriggerResult:
        """Thin wrapper around `_execute_core` that stamps observability
        timestamps onto `new_provider_state` regardless of which branch
        of the core returned. The `/probe` endpoint reads these so the
        operator can tell "handler ran recently" from "dispatched but
        the handler never reached terminal state."

        This wrapper is NOT a new abstraction — it's a single tail
        stamp run on every result. The eight return sites in the core
        would otherwise each need the same three lines duplicated.
        """
        result = await self._execute_core(trigger, events, db)
        # Mirror `status` into a stable "last_handler_status" string the
        # frontend / probe surface can read without knowing the enum.
        # The runner already maps `status` to `Trigger.last_status` with
        # different semantics; this field is separate, observability-only.
        new_ps = dict(result.new_provider_state or {})
        new_ps["last_handler_completed_at"] = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        new_ps["last_handler_status"] = result.status
        if result.error_class:
            new_ps["last_handler_error_class"] = result.error_class
        else:
            # Clear stale error from prior failure so the probe doesn't
            # report a long-fixed error as current.
            new_ps.pop("last_handler_error_class", None)
        result.new_provider_state = new_ps
        return result

    async def _execute_core(
        self,
        trigger: Any,
        events: list[Any],
        db: AsyncSession,
    ) -> TriggerResult:
        # ── 0. Synthetic test-fire short-circuit ──────────────────────
        # When EVERY event in the batch is synthetic (idempotency_key
        # starts with "test:"), bypass the LLM / MCP / broadcast chain
        # entirely and write a deterministic "wiring works" message
        # straight to Day-as-Chat. The Test button's job is to answer
        # "is the trigger plumbing alive?" — it must not also be a
        # health check for OpenAI/Anthropic, the MCP client, or the
        # connector vault. Bug 2 in the 2026-05-16 fix series:
        # previously test fires went through call_system_llm and
        # silently failed when OpenAI was rate-limited / down / mis-keyed,
        # producing the misleading "5 fires, all Failed" UI state.
        #
        # PR #49 cutover: events are now ``BuildJob`` rows directly, not
        # the legacy ``TriggerEvent`` shape. The legacy
        # ``event_dedupe_id`` column maps 1:1 to ``BuildJob.idempotency_key``
        # (see ``app/api/triggers_inbound.py::_idempotent_insert`` and
        # ``app/api/triggers.py`` test-fire path — both write the Gmail
        # message id, or the synthetic "test:<uuid>", into
        # ``idempotency_key``). Reading ``event_dedupe_id`` on a BuildJob
        # raises ``AttributeError`` — that was the
        # ``all_retries_exhausted: AttributeError("'BuildJob' object has
        # no attribute 'event_dedupe_id'")`` failure surfaced in
        # production Gmail trigger fires.
        if events and all(
            (getattr(ev, "idempotency_key", None) or "").startswith("test:")
            for ev in events
        ):
            return await self._do_test_fire(trigger, events, db)

        mcp = self._mcp_client
        if mcp is None:
            return TriggerResult(
                status="failed",
                per_event_status={e.id: "failed" for e in events},
                error_class="no_mcp_client",
                error_detail="TriggerRunner did not provide an MCP client",
            )

        # ── 1. Fetch each message ──
        try:
            fetched, fetch_errors = await self._fetch_all(mcp, events)
        except _ReauthRequired as e:
            # Whole batch fails the same way — auth is broken. Fan out
            # a "reconnect Gmail" card on every delivery_channel the
            # trigger is configured for so the user actually finds out
            # (not just on website). Dedupe at process level so a burst
            # of failed pushes doesn't spam.
            await self._notify_reauth(trigger, reason="gmail_token_revoked")
            return TriggerResult(
                status="skipped_reauth",
                per_event_status={ev.id: "failed" for ev in events},
                error_class="reauth_required",
                error_detail=str(e)[:300],
            )
        except _ToolMissing as e:
            await self._notify_reauth(trigger, reason="gmail_tool_missing")
            return TriggerResult(
                status="skipped_reauth",
                per_event_status={ev.id: "failed" for ev in events},
                error_class="tool_missing",
                error_detail=str(e)[:300],
            )

        # ── 2. Apply filters — drop events whose message doesn't match ──
        kept: list[_FetchedEmail] = []
        per_event_status: dict[str, str] = {}
        for ev_id, err in fetch_errors.items():
            per_event_status[ev_id] = "failed"

        for fe in fetched:
            matched, reason = matches_filter(fe.raw_message, trigger.filter_json)
            if not matched:
                per_event_status[fe.event_id] = "skipped_filter"
                logger.info(
                    "[trigger_email] filter_drop trigger_id=%s event_id=%s reason=%s",
                    trigger.id, fe.event_id, reason,
                )
                continue
            kept.append(fe)

        if not kept:
            # All filtered out OR all events failed to fetch — the handler
            # ran clean but did NOT deliver a message anywhere. Distinguish
            # this from "delivered a real summary" so the runner doesn't
            # false-promote the trigger to last_status='active'. Bug 1 in
            # the 2026-05-16 fix series: returning status="success" here
            # caused triggers with zero real successful fires to show a
            # green "Active" pill alongside a list of failed events.
            had_fetch_failure = bool(fetch_errors)
            return TriggerResult(
                status="success_empty",
                per_event_status=per_event_status,
                error_class="all_fetch_failed" if had_fetch_failure else "all_filtered_out",
                error_detail=(
                    "Every event in the batch failed to fetch from Gmail."
                    if had_fetch_failure
                    else "Every event matched a filter exclusion rule."
                ),
                metrics={
                    "fetched": len(fetched),
                    "filtered_kept": 0,
                    "fetch_errors": len(fetch_errors),
                },
            )

        # ── 3. Dispatch by action ──
        action = (trigger.action or "").strip()
        try:
            if action == "summarize_and_post":
                msg_id, model = await self._do_summarize_and_post(
                    trigger, kept, db,
                )
            elif action == "notify_only":
                msg_id, model = await self._do_notify_only(
                    trigger, kept, db,
                )
            elif action == "forward_to_telegram":
                msg_id, model = await self._do_forward_to_telegram(
                    trigger, kept, db,
                )
            else:
                # Unknown action — schema-level enum should have caught
                # this. Fail safely so the trigger flips to last_status=failed
                # and operator can see it in Mission Control.
                return TriggerResult(
                    status="failed",
                    per_event_status={ev.id: "failed" for ev in events},
                    error_class="unknown_action",
                    error_detail=f"trigger.action={action!r} not in canonical set",
                )
        except Exception as e:
            logger.exception(
                "[trigger_email] action_failed trigger_id=%s action=%s err=%s",
                trigger.id, action, e,
            )
            return TriggerResult(
                status="failed",
                per_event_status={ev.id: "failed" for ev in events},
                error_class=type(e).__name__,
                error_detail=str(e)[:300],
            )

        # ── 4. Mark every kept event success, point at the Message ──
        for fe in kept:
            per_event_status[fe.event_id] = "success"

        return TriggerResult(
            status="success",
            per_event_status=per_event_status,
            summary_message_id=msg_id,
            metrics={
                "fetched": len(fetched),
                "filtered_kept": len(kept),
                "action": action,
                "model": model,
                "coalesced_count": len(events) - 1 if len(events) > 1 else 0,
            },
        )

    # ── Synthetic test fire (Test button) ─────────────────────────

    async def _do_test_fire(
        self, trigger: Any, events: list[Any], db: AsyncSession,
    ) -> TriggerResult:
        """Deterministic write path for events from the Test button.

        Bypasses MCP, the LLM, and the rate limiter so a "Test" click
        answers exactly one question — is the trigger row wired to the
        Day-as-Chat writer? — and gives the user a real signal that's
        unaffected by external service health. The result is reported
        with ``status="test_success"`` so the runner can keep the
        trigger's ``last_status`` honest: a test pass proves the
        plumbing but does NOT prove a real Gmail event would deliver.
        """
        msg_body = (
            "**Trigger wiring check — passed.**\n\n"
            f"Synthetic test fire for trigger "
            f"**{trigger.name or 'Gmail trigger'}**. No real Gmail "
            "message was fetched — this run only confirms the agent "
            "can write to Day-as-Chat on this trigger's behalf and "
            f"fan out to {', '.join((trigger.config_json or {}).get('delivery_channels') or ['website'])}.\n\n"
            "A real inbound email will deliver the actual summary."
        )

        from app.db.database import async_session_maker
        writer = self._writer or write_trigger_message
        broadcaster = self._broadcaster or broadcast_trigger_message

        try:
            async with async_session_maker() as wdb:
                msg_id, day_chat_id = await writer(
                    wdb,
                    user_id=trigger.user_id,
                    content=msg_body,
                    source=self.kind,
                    trigger_id=trigger.id,
                    title=f"{trigger.name or 'Email trigger'} — wiring check",
                    model_used=None,
                )
        except Exception as e:
            logger.exception(
                "[trigger_email] test_fire_write_failed trigger_id=%s err=%s",
                trigger.id, e,
            )
            return TriggerResult(
                status="failed",
                per_event_status={ev.id: "failed" for ev in events},
                error_class=type(e).__name__,
                error_detail=f"Day-as-Chat write failed: {e}"[:300],
            )

        delivery_channels = list(
            (trigger.config_json or {}).get("delivery_channels") or ["website"]
        )
        try:
            await broadcaster(
                trigger.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=msg_body,
                model_used=None,
                delivery_channels=delivery_channels,
                trigger_name=trigger.name or trigger.kind,
            )
        except Exception as e:
            # Broadcast failure is non-fatal for the test: the message
            # landed in Day-as-Chat, which is what the test promises.
            # Telegram/WhatsApp fan-out can fail later without making the
            # wiring check itself a lie.
            logger.warning(
                "[trigger_email] test_fire_broadcast_failed trigger_id=%s err=%s",
                trigger.id, e,
            )

        return TriggerResult(
            status="test_success",
            per_event_status={ev.id: "success" for ev in events},
            summary_message_id=msg_id,
            metrics={
                "fetched": 0,
                "filtered_kept": 0,
                "action": "test_fire",
                "model": None,
                "test_event_count": len(events),
            },
        )

    # ── Reauth notice (channel-aware) ─────────────────────────────

    async def _notify_reauth(self, trigger: Any, *, reason: str) -> None:
        """Fan out a 'reconnect Gmail' card on every configured
        delivery_channel (website + Telegram + WhatsApp). Deduped per
        (user_id, trigger_id) for 24h so a burst of failing pushes
        doesn't spam.

        Failures here are swallowed — the trigger run is already in
        the failure path; we won't compound it by raising over a
        notification problem.
        """
        import time
        key = (str(trigger.user_id), str(trigger.id))
        last = _REAUTH_NOTICE_DEDUPE.get(key, 0.0)
        now = time.time()
        if (now - last) < _REAUTH_NOTICE_TTL_SEC:
            logger.info(
                "[trigger_email] reauth_notice deduped trigger_id=%s "
                "user_id=%s age_sec=%d",
                trigger.id, str(trigger.user_id)[:8], int(now - last),
            )
            return

        content = _REAUTH_NOTICE_CARD.format(
            trigger_name=(trigger.name or "Gmail trigger"),
        )
        delivery_channels = list(
            (trigger.config_json or {}).get("delivery_channels") or ["website"]
        )

        # Persist a Day-as-Chat message (website channel) AND fan out
        # to any extras. We reuse the trigger writer so the card lands
        # in the same conversation the user expects trigger output in.
        from app.db.database import async_session_maker
        try:
            async with async_session_maker() as db:
                writer = self._writer or write_trigger_message
                broadcaster = self._broadcaster or broadcast_trigger_message
                msg_id, day_chat_id = await writer(
                    db,
                    user_id=trigger.user_id,
                    content=content,
                    source=self.kind,
                    trigger_id=trigger.id,
                    title=f"{trigger.name or 'Gmail trigger'} — reconnect needed",
                    model_used=None,
                )
                try:
                    await broadcaster(
                        trigger.user_id,
                        message_id=msg_id,
                        day_chat_id=day_chat_id,
                        source=self.kind,
                        content=content,
                        model_used=None,
                        delivery_channels=delivery_channels,
                        trigger_name=trigger.name or trigger.kind,
                    )
                except Exception as e:
                    logger.warning(
                        "[trigger_email] reauth_notice broadcast failed: %s", e,
                    )
        except Exception as e:
            logger.warning(
                "[trigger_email] reauth_notice persist failed reason=%s: %s",
                reason, e,
            )
            return

        _REAUTH_NOTICE_DEDUPE[key] = now
        logger.info(
            "[trigger_email] reauth_notice sent trigger_id=%s user_id=%s "
            "channels=%s reason=%s",
            trigger.id, str(trigger.user_id)[:8],
            ",".join(delivery_channels), reason,
        )

    # ── Fetch ─────────────────────────────────────────────────────

    async def _fetch_all(
        self, mcp: Any, events: list[Any],
    ) -> tuple[list[_FetchedEmail], dict[str, str]]:
        """Fetch every message in the batch. Returns (fetched_list,
        per-event errors). The errors dict carries event_id → error
        class for events that failed to fetch (network, deleted, etc.)
        — the caller marks those failed and continues with the rest.

        Raises _ReauthRequired / _ToolMissing to short-circuit the
        whole batch if the connector is unauthorised.
        """
        fetched: list[_FetchedEmail] = []
        errors: dict[str, str] = {}

        for ev in events:
            # PR #49 cutover: ``ev`` is a ``BuildJob`` row, not the
            # legacy ``TriggerEvent`` shape. The Gmail message id lives
            # in ``BuildJob.idempotency_key`` — set by both
            # ``triggers_inbound._idempotent_insert`` (production Pub/Sub
            # path) and ``triggers.py`` test-fire (`test:<uuid>` synthetic
            # path). ``external_payload`` was the legacy carrier and was
            # dropped during the cutover; ``config_json`` is the new
            # opaque-per-row carrier but the inbound endpoint doesn't
            # populate it for trigger fires (idempotency_key is enough).
            #
            # Reading ``ev.external_payload`` or ``ev.event_dedupe_id``
            # on a BuildJob raises ``AttributeError`` — which surfaced as
            # ``all_retries_exhausted: AttributeError(...)`` on every
            # Gmail trigger fire in production until this fix.
            gmail_id = (
                str(getattr(ev, "idempotency_key", None) or "")
            ).strip()
            # Defensive: config_json may carry an explicit
            # ``gmail_message_id`` in the future (e.g., if the inbound
            # endpoint starts persisting external_payload there). Prefer
            # that when present.
            cfg = getattr(ev, "config_json", None)
            if isinstance(cfg, dict):
                explicit = str(cfg.get("gmail_message_id") or "").strip()
                if explicit:
                    gmail_id = explicit
            if not gmail_id:
                errors[ev.id] = "no_gmail_id"
                continue

            # ── Test-fire short-circuit ─────────────────────────────
            # Events whose dedupe_id starts with "test:" are synthetic
            # — created by `POST /api/triggers/{id}/test`. The dedupe
            # id is a uuid hex, NOT a real Gmail message id, so the MCP
            # fetch would 404 and the test would falsely look "Failed."
            # Inject a hand-built _FetchedEmail so the whole pipeline
            # (filter, summarise, post) runs against synthetic data and
            # the user actually sees a "test event" message in chat —
            # which is what the success banner promises.
            if gmail_id.startswith("test:"):
                fetched.append(_synthetic_test_email(ev.id, gmail_id))
                continue

            try:
                # FastMCP Client requires ``async with`` to establish
                # the streamable-HTTP session before call_tool can
                # round-trip. Without this wrapper every call_tool
                # raised a "session not started" exception that got
                # caught generically below as a fetch failure — the
                # sibling email_briefing handler at routines/
                # email_briefing_handler.py:469 has had this wrapper
                # since launch; the trigger handler missed it. Caught
                # 2026-05-18 after fixing the message_id param and
                # still seeing every event fail with
                # all_fetch_failed AND no `tool_called` event landing
                # on the platform (confirming the call never left the
                # agent). Param ``message_id`` matches the connector
                # provider's tool_input schema (was previously ``id``
                # in the same regression).
                async with mcp:
                    call_result = await mcp.call_tool(
                        "gmail__get_message",
                        {"message_id": gmail_id},
                    )
                data = _unpack_ok(call_result, "gmail__get_message")
            except _ReauthRequired:
                raise
            except _ToolMissing:
                raise
            except Exception as e:
                # Could be a permanent 404 (message deleted before
                # fetch) or a transient API hiccup. Mark this event
                # failed; the rest of the batch continues.
                logger.warning(
                    "[trigger_email] fetch_failed event_id=%s gmail_id=%s err=%s",
                    ev.id, gmail_id, e,
                )
                errors[ev.id] = type(e).__name__
                continue

            # The Gmail connector returns a pre-flattened envelope:
            # {headers: {From,Subject,...}, body, snippet, labelIds}.
            # Raw Gmail REST is {payload.headers: [{name,value}], ...}.
            # Detect which one and read accordingly.
            raw_headers = data.get("headers")
            if isinstance(raw_headers, dict):
                headers = dict(raw_headers)
                body_text = (data.get("body") or "").strip() \
                    or (data.get("snippet") or "")
            else:
                headers = _flatten_headers(data)
                body_text = _extract_body(data) or data.get("snippet") or ""

            fetched.append(_FetchedEmail(
                event_id=ev.id,
                gmail_id=gmail_id,
                headers=headers,
                snippet=data.get("snippet") or "",
                body=body_text,
                labels=list(data.get("labelIds") or []),
                raw_message=data,
            ))

        return fetched, errors

    # ── Actions ───────────────────────────────────────────────────

    async def _do_summarize_and_post(
        self, trigger: Any, emails: list[_FetchedEmail], db: AsyncSession,
    ) -> tuple[str, str]:
        """LLM summary action. Single email → tight summary. Multi
        (coalesced burst) → digest."""
        llm = self._llm_fn
        if llm is None:
            from app.services.internal_llm import call_system_llm
            llm = call_system_llm

        # Build the prompt
        if len(emails) == 1:
            user_msg = _format_one_for_llm(emails[0])
        else:
            user_msg = _format_batch_for_llm(emails)

        cfg = trigger.config_json or {}
        # Default-cheap policy (2026-05-13 cost audit): when no explicit
        # `model` is configured, route per-email summarization to
        # gpt-4o-mini ($0.15/1M input) instead of falling through to
        # the user's chat-model default. A busy inbox can fire this
        # 50-100×/day; on GPT-5.5 that's $1-2/day per active user just
        # for triage. gpt-4o-mini is plenty for "summarize this email +
        # extract action items". Power users wanting Sonnet/Opus set
        # `config_json.model` on the trigger explicitly.
        model_choice = (cfg.get("model") or "").strip() or "gpt-4o-mini"

        text = await llm(
            user_id=trigger.user_id,
            operation_type=_OPERATION_TYPE,
            model=model_choice,
            max_tokens=_SUMMARY_MAX_TOKENS,
            system=_SUMMARIZE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
            timeout=120,
        )
        if not text:
            raise RuntimeError(
                "internal_llm returned None (timeout / auth / parse failure)"
            )

        # Output-guard at the LLM boundary. The system prompt tells the
        # model to only emit "(no body)" when the body is literally
        # empty, but gpt-4o-mini still occasionally bails on short or
        # cryptic content (e.g. immigration acronyms like "pgwp" /
        # "workpwrmit" — caught 2026-05-18, gmail msg 19e3cea85b2c9bcf
        # where the body was the literal string the user had typed but
        # the model returned "(no body)"). When that happens, swap in
        # a deterministic fallback that quotes the sender's actual text
        # so the user always sees what they wrote. Catches single-email
        # case; batch case is rarer and the per-email gist guards itself.
        text = _guard_summary_against_no_body(text, emails)

        model_used = model_choice

        # Wrap the LLM body with a Gmail-flavoured header so the chat
        # card reads like a notification, not a generic agent reply.
        # The frontend's markdown renderer turns this into the same
        # visual structure as the "Latest Gmail" cards from the manual
        # fetch path. Without this header the trigger output was just
        # raw bullet text starting with "(unknown) →" which gave the
        # user no signal it was a Gmail event. Improved 2026-05-18 from
        # production feedback.
        content = _compose_email_chat_card(emails, text)

        return await self._persist_message(
            trigger=trigger, db=db,
            content=content,
            model_used=model_used,
            title_suffix=_compose_title_suffix(emails),
        )

    async def _do_notify_only(
        self, trigger: Any, emails: list[_FetchedEmail], db: AsyncSession,
    ) -> tuple[str, str]:
        """Raw notification — no LLM. One card per email when batched."""
        if len(emails) == 1:
            content = _format_one_for_notify(emails[0])
        else:
            content = (
                f"**You got {len(emails)} new emails:**\n\n"
                + "\n\n".join(_format_one_for_notify(e) for e in emails)
            )

        return await self._persist_message(
            trigger=trigger, db=db,
            content=content,
            model_used=None,
            title_suffix=_compose_title_suffix(emails),
        )

    async def _do_forward_to_telegram(
        self, trigger: Any, emails: list[_FetchedEmail], db: AsyncSession,
    ) -> tuple[str, str]:
        """Summarise (Day-as-Chat) + force Telegram fan-out. Reuses the
        summarize_and_post code path but appends 'telegram' to the
        delivery_channels passed to the broadcaster."""
        msg_id, model_used = await self._do_summarize_and_post(
            trigger, emails, db,
        )
        # The broadcaster reads delivery_channels from the trigger config —
        # we don't need to override here since _persist_message already
        # invoked it. forward_to_telegram is mostly a UX label that says
        # "ensure telegram is in delivery_channels at create time" —
        # which the API + skill enforce. If a misconfigured row arrived
        # without telegram in delivery_channels, fan it out now anyway.
        existing_channels = (trigger.config_json or {}).get("delivery_channels") or []
        if "telegram" not in existing_channels:
            try:
                from app.agent.routines.channel_dispatcher import (
                    deliver_to_extra_channels,
                )
                from app.db.database import async_session_maker
                # We need the summary content — re-derive (or hold it
                # via a closure). The previous call already wrote the
                # message; the simplest path is to query it back.
                from app.db.models import Message
                row = await db.get(Message, msg_id)
                content = row.content if row else "(message vanished)"
                await deliver_to_extra_channels(
                    user_id=trigger.user_id,
                    delivery_channels=["telegram"],
                    routine_name=trigger.name or trigger.kind,
                    content=content,
                    db_session_maker=async_session_maker,
                )
            except Exception as e:
                logger.warning(
                    "[trigger_email] forward_to_telegram fanout failed: %s", e,
                )
        return msg_id, model_used

    # ── Persistence ───────────────────────────────────────────────

    async def _persist_message(
        self,
        *,
        trigger: Any,
        db: AsyncSession,
        content: str,
        model_used: Optional[str],
        title_suffix: str,
    ) -> tuple[str, str]:
        writer = self._writer or write_trigger_message
        broadcaster = self._broadcaster
        if broadcaster is None:
            broadcaster = broadcast_trigger_message

        title = f"{trigger.name or 'Email trigger'} — {title_suffix}"
        msg_id, day_chat_id = await writer(
            db,
            user_id=trigger.user_id,
            content=content,
            source=self.kind,
            trigger_id=trigger.id,
            title=title,
            model_used=model_used,
        )

        delivery_channels = list(
            (trigger.config_json or {}).get("delivery_channels") or ["website"]
        )
        try:
            await broadcaster(
                trigger.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=content,
                model_used=model_used,
                delivery_channels=delivery_channels,
                trigger_name=trigger.name or trigger.kind,
            )
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("[trigger_email] broadcast failed: %s", e)

        return msg_id, (model_used or "n/a")


# ── Formatters ────────────────────────────────────────────────────────


def _h(headers: dict, key: str, default: str = "") -> str:
    """Case-insensitive header lookup. Gmail's connector provider
    returns headers with the canonical capitalised RFC 5322 names
    (``From``, ``Subject``, ``Date``); the handler used to read with
    lowercase keys, so every sender/subject came back as the default
    and the LLM happily included ``(unknown)`` in the summary. Caught
    2026-05-18 once the end-to-end pipeline finally produced a real
    chat card and Nariman's barbecue email rendered as
    ``(unknown) → Informing Ali``. Fixing this in one helper makes the
    fetch + LLM + notify paths agree."""
    for k in (key, key.lower(), key.title(), key.upper()):
        if k in headers:
            return headers[k] or default
    return default


def _format_one_for_llm(e: _FetchedEmail) -> str:
    sender = _h(e.headers, "From", "(unknown sender)")
    subject = _h(e.headers, "Subject", "(no subject)")
    date = _h(e.headers, "Date", "")
    body = (e.body or e.snippet or "").strip()
    if len(body) > 2500:
        body = body[:2500] + "…"
    return (
        f"From: {sender}\nSubject: {subject}\nDate: {date}\n\n{body}"
    )


def _format_batch_for_llm(emails: list[_FetchedEmail]) -> str:
    blocks: list[str] = []
    for e in emails:
        sender = _h(e.headers, "From", "(unknown sender)")
        subject = _h(e.headers, "Subject", "(no subject)")
        body = (e.body or e.snippet or "").strip()
        if len(body) > 800:
            body = body[:800] + "…"
        blocks.append(
            f"--- email id={e.gmail_id}\n"
            f"From: {sender}\nSubject: {subject}\n\n{body}"
        )
    return "\n\n".join(blocks)


def _format_one_for_notify(e: _FetchedEmail) -> str:
    sender = _h(e.headers, "From", "(unknown sender)")
    subject = _h(e.headers, "Subject", "(no subject)")
    snippet = (e.snippet or "").strip()
    if len(snippet) > 200:
        snippet = snippet[:200] + "…"
    return f"**{sender}**\n*{subject}*\n\n{snippet or '_(empty body)_'}"


def _compose_title_suffix(emails: list[_FetchedEmail]) -> str:
    if len(emails) == 1:
        subj = _h(emails[0].headers, "Subject", "(no subject)")
        if len(subj) > 60:
            subj = subj[:60] + "…"
        return subj
    return f"{len(emails)} new emails"


def _compose_email_chat_card(
    emails: list[_FetchedEmail], summary_body: str,
) -> str:
    """Wrap the LLM-produced summary with a recognisable Gmail header
    card so the chat surface renders it as a notification, not a
    formless agent reply. Single email → "📧 **Sender** — Subject"
    line, then the summary body. Batch → "📧 **N new emails**" line,
    then a one-line preview per email, then the summary. Markdown is
    intentionally minimal — the day-chat renderer is what styles it.

    Why a header at all: the trigger card the user saw on first
    end-to-end fire ("(unknown) → Informing Ali…") had no visible
    indication it was a Gmail event, no Gmail glyph, no sender name.
    Pre-pending a structured header gives the renderer the
    information it needs without changing the storage schema, and the
    LLM is told (in the system prompt) NOT to repeat any of this.
    """
    if not emails:
        return summary_body
    if len(emails) == 1:
        e = emails[0]
        sender = _sender_display(_h(e.headers, "From", "(unknown sender)"))
        subject = _h(e.headers, "Subject", "(no subject)").strip() or "(no subject)"
        return (
            f"📧 **{sender}** — _{subject}_\n\n"
            f"{summary_body.strip()}"
        )
    # Batch: small one-line preview per email above the bulk summary.
    preview = "\n".join(
        f"• **{_sender_display(_h(e.headers, 'From', '(unknown sender)'))}**"
        f" — _{(_h(e.headers, 'Subject', '(no subject)') or '(no subject)').strip()}_"
        for e in emails
    )
    return (
        f"📧 **{len(emails)} new emails**\n\n"
        f"{preview}\n\n"
        f"{summary_body.strip()}"
    )


_NO_BODY_OUTPUTS = {
    "(no body)",
    "no body",
    "(no body.)",
    "no body.",
    "(empty)",
    "the body is empty",
    "the email body is empty",
    "there is no body",
}


def _guard_summary_against_no_body(
    summary: str, emails: list[_FetchedEmail],
) -> str:
    """If the LLM emitted a "(no body)"-shaped response but we hold
    real body text, replace the response with a deterministic quote of
    what the sender wrote. Trust the source of truth (the fetched
    Gmail body) over the model's interpretation.

    Single-email only — batch summaries get one combined response and
    almost never trip this case in practice; if they do, the user can
    still see per-email previews above the summary block.
    """
    if not emails or len(emails) != 1:
        return summary
    stripped = (summary or "").strip().lower().rstrip(".!")
    if stripped not in _NO_BODY_OUTPUTS:
        return summary
    e = emails[0]
    body_text = (e.body or e.snippet or "").strip()
    if not body_text:
        # LLM was right — body really is empty. Pass through.
        return summary
    # Trim to a one-card-sized excerpt; the trigger surface is a
    # notification, not a full read pane.
    excerpt = body_text if len(body_text) <= 400 else body_text[:400] + "…"
    return excerpt


def _sender_display(raw: str) -> str:
    """Strip the email-in-angle-brackets to leave a clean name when
    Gmail's From header is the canonical ``Name <addr@host>`` form.
    Fallback: address-as-name if there's no display name."""
    raw = (raw or "").strip()
    if "<" in raw and ">" in raw:
        name = raw.split("<")[0].strip().strip('"').strip("'")
        if name:
            return name
        addr = raw.split("<")[1].split(">")[0].strip()
        return addr or raw
    return raw


# ── Gmail body extraction ─────────────────────────────────────────────


def _extract_body(message: dict) -> Optional[str]:
    """Walk the Gmail payload tree and pull the text/plain body.
    Falls back to text/html (stripped) if no plain part is present.
    Returns None when no readable body found — caller falls back to
    snippet for substring matching."""
    payload = message.get("payload") or {}
    plain = _walk_for_mime(payload, "text/plain")
    if plain:
        return plain
    html = _walk_for_mime(payload, "text/html")
    if html:
        # Naive HTML strip — good enough for filter substring + LLM
        # context. Day-archival has a fancier path; we don't need it.
        import re
        return re.sub(r"<[^>]+>", " ", html)
    return None


def _walk_for_mime(part: dict, mime: str) -> Optional[str]:
    if part.get("mimeType") == mime:
        body = part.get("body") or {}
        data = body.get("data")
        if data:
            import base64
            try:
                # Gmail uses base64url
                padded = data + "=" * (-len(data) % 4)
                padded = padded.replace("-", "+").replace("_", "/")
                return base64.b64decode(padded).decode(
                    "utf-8", errors="replace"
                )
            except Exception:
                return None
    for sub in part.get("parts") or []:
        found = _walk_for_mime(sub, mime)
        if found is not None:
            return found
    return None


# ── Auto-register at import ───────────────────────────────────────────
# The handler is a singleton; mcp_client + writer are wired by the
# runner via attribute assignment after construction. Same pattern
# as the routines registry.

_handler_singleton: Optional[EmailReceivedHandler] = None


def get_handler() -> EmailReceivedHandler:
    global _handler_singleton
    if _handler_singleton is None:
        _handler_singleton = EmailReceivedHandler()
    return _handler_singleton


def _auto_register():
    from .registry import register_handler
    register_handler(get_handler())


_auto_register()
