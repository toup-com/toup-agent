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
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
_OPERATION_TYPE = "system.trigger.email_received"


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


_SUMMARIZE_SYSTEM_PROMPT = """You are summarising a fresh inbound email (or short burst of them) for the user.

Rules:
- Open with the absolute essentials: who, what, why it matters.
- For one email: 2–4 sentence summary + an explicit "Action: …" line ONLY if a reply / deadline / approval is being requested.
- For multiple emails: open with "You got N new emails." then one bullet per email (sender → one-line gist + action if any).
- Skip newsletters, promotions, automated noise. Note them as "(N newsletters skipped)".
- ≤200 words for one email, ≤350 words for a burst. Hard cap 500.
- Markdown supported — use **bold** for sender names + ⚑ for boss/family/financial/security urgency.
- No greeting, no sign-off, no "Here is a summary".

Never invent details. If the body was empty, say so."""


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
            # All filtered out — return success on the batch (nothing to do,
            # but the handler ran clean). Per-event statuses already set.
            return TriggerResult(
                status="success",
                per_event_status=per_event_status,
                metrics={"fetched": len(fetched), "filtered_kept": 0},
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
            payload = ev.external_payload if hasattr(ev, "external_payload") else {}
            # The inbound endpoint stores the payload on the row via
            # `external_payload` — but our DB column is named
            # event_dedupe_id (the gmail message id). Either source
            # works; prefer the explicit payload, fall back to dedupe.
            gmail_id = ""
            if isinstance(payload, dict):
                gmail_id = str(payload.get("gmail_message_id") or "").strip()
            if not gmail_id:
                gmail_id = (ev.event_dedupe_id or "").strip()
            if not gmail_id:
                errors[ev.id] = "no_gmail_id"
                continue

            try:
                call_result = await mcp.call_tool(
                    "gmail__get_message",
                    {"id": gmail_id, "format": "full"},
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

            headers = _flatten_headers(data)
            fetched.append(_FetchedEmail(
                event_id=ev.id,
                gmail_id=gmail_id,
                headers=headers,
                snippet=data.get("snippet") or "",
                body=_extract_body(data) or data.get("snippet") or "",
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

        model_used = model_choice

        return await self._persist_message(
            trigger=trigger, db=db,
            content=text,
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


def _format_one_for_llm(e: _FetchedEmail) -> str:
    sender = e.headers.get("from", "(unknown)")
    subject = e.headers.get("subject", "(no subject)")
    date = e.headers.get("date", "")
    body = (e.body or e.snippet or "").strip()
    if len(body) > 2500:
        body = body[:2500] + "…"
    return (
        f"From: {sender}\nSubject: {subject}\nDate: {date}\n\n{body}"
    )


def _format_batch_for_llm(emails: list[_FetchedEmail]) -> str:
    blocks: list[str] = []
    for e in emails:
        sender = e.headers.get("from", "(unknown)")
        subject = e.headers.get("subject", "(no subject)")
        body = (e.body or e.snippet or "").strip()
        if len(body) > 800:
            body = body[:800] + "…"
        blocks.append(
            f"--- email id={e.gmail_id}\n"
            f"From: {sender}\nSubject: {subject}\n\n{body}"
        )
    return "\n\n".join(blocks)


def _format_one_for_notify(e: _FetchedEmail) -> str:
    sender = e.headers.get("from", "(unknown)")
    subject = e.headers.get("subject", "(no subject)")
    snippet = (e.snippet or "").strip()
    if len(snippet) > 200:
        snippet = snippet[:200] + "…"
    return f"**{sender}**\n*{subject}*\n\n{snippet or '_(empty body)_'}"


def _compose_title_suffix(emails: list[_FetchedEmail]) -> str:
    if len(emails) == 1:
        subj = emails[0].headers.get("subject", "(no subject)")
        if len(subj) > 60:
            subj = subj[:60] + "…"
        return subj
    return f"{len(emails)} new emails"


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
