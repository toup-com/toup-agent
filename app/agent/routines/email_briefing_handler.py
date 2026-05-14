"""Email briefing handler — routine kind #1.

Fetches new Gmail messages since the user's last briefing, summarizes
them via Claude Haiku, posts the summary into Day-as-Chat as
role=assistant + channel=routine + source=email_briefing.

Contract is `RoutineHandler` (see base_handler.py): handler returns a
`RoutineResult`; runner owns retry, run-row finalization, and
nudge-on-failure. The handler does NOT touch `routine_runs` rows.

Key design choices:
- Pre-flight is the first MCP call. If Gmail's vault status is
  reauth_required / provider_down the platform dispatcher returns a
  structured `kind=reauth_required` envelope which we recognize and
  short-circuit on; no separate pre-flight check is needed.
- Bootstrap (no watermark): query `newer_than:1d`. Steady state: use
  hours since `last_processed_internal_date`, capped at 168h (one
  week) so a long agent outage doesn't pull a junk drawer's worth.
- Fetch is serialized (`gmail__get_message` per message id), capped at
  50 emails per briefing to stay under Gmail's 250 quota-unit/sec
  ceiling without parallel fan-out. >50 → summarize top 50, surface
  "+N more" in the body.
- Empty result still posts a "no new emails since last briefing" line so
  the user has the signal the routine ran. Watermark does NOT advance on
  empty (we don't have a max(internal_date) to advance to).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .base_handler import RoutineResult


logger = logging.getLogger(__name__)


# Conservative summary cap. 300-word target ≈ 400 tokens; ceiling at 1200
# gives margin without inviting LLM ramble. Day-archival uses 1800 for a
# longer summary; we're tighter.
_SUMMARY_MAX_TOKENS = 1200

# Pre-refactor we hard-pinned `claude-haiku-4-5-20251001` here. That was a
# bug: bundle-mode tenants whose chat uses GPT-5.5 would suddenly find
# their morning brief being written by an Anthropic model on credentials
# the rest of the stack stopped renewing. The model now comes from the
# `call_system_llm` resolver — same path the chat uses — with an optional
# per-routine override via `routine.config_json.model` for power users
# who genuinely want to pin "always Haiku, my brief is text-heavy" or
# similar.

# Gmail-side caps.
_MAX_EMAILS_PER_BRIEFING = 50
_BOOTSTRAP_WINDOW = "newer_than:1d"
_STEADY_WINDOW_CAP_HOURS = 168  # 7 days

# operation_type prefix for internal_llm system-tagged budgeting.
_OPERATION_TYPE = "system.routine.email_briefing"

SYSTEM_PROMPT = """You write a concise morning email briefing. The user will read this on their phone over coffee. Rules:

- Open with a one-line summary of the inbox state ("12 new emails overnight, 3 need a reply today").
- Group by sender/thread, not by time order. One bullet per thread.
- Surface action items: replies needed, deadlines today/this week, account/security alerts, calendar invites, payments.
- Flag priority: ⚑ for boss/family/financial/security; nothing for routine.
- Skip newsletters, promotions, automated noise. Mention them only as "(N newsletters skipped)".
- ≤300 words. Hard cap 500. Markdown supported — use **bold** and bullets, no headings.
- No greeting, no sign-off, no preamble. Start with the one-line summary.

If there are no new emails, output exactly: "No new emails since the last briefing."
"""


# Alternate prompt for `mode=latest_n` routines (Ticket 7, 2026-05-13).
# The default SYSTEM_PROMPT is hard-wired to "new since last briefing"
# semantics and will say "No new emails since the last briefing" on a
# quiet day even when latest-N emails were fetched successfully. A
# user who set up "daily latest 5 Gmail" wants the most recent five
# every day, not "nothing new since yesterday's run."
SYSTEM_PROMPT_LATEST_N = """You're showing the user their latest emails on a scheduled morning briefing. Rules:

- Open with one line stating how many emails you're showing ("Here are your latest 5 emails:").
- Then list them in reverse-chronological order — one block per email:
    • **Sender — Subject**
    • Date (use the user's local format).
    • 1-2 line gist of the body. Surface action items, deadlines,
      security alerts, payments, calendar invites.
- Flag priority with ⚑ for boss/family/financial/security; nothing for routine.
- Skip newsletter/promotional noise only if you have MORE than the
  requested count — never pad with noise to hit a number.
- ≤300 words total. Markdown supported — use **bold** and bullets, no
  headings.
- DO NOT say "no new emails since the last briefing" — these are the
  latest, not the new. If the inbox is genuinely empty, say so plainly.
"""


@dataclass
class _Email:
    id: str
    thread_id: str
    headers: dict
    snippet: str
    body: str
    internal_date: Optional[int]  # Gmail's milliseconds-since-epoch


class _ReauthRequired(Exception):
    """Sentinel raised when an MCP call returns kind=reauth_required.
    The handler converts this into RoutineResult(status=skipped_reauth)
    so the runner can post the reconnect nudge."""


class _ToolMissing(Exception):
    """Gmail tool wasn't in the tools/list. Either Gmail is disconnected
    or the connector code hasn't shipped to this container yet.
    Treated identically to reauth — user needs to (re)connect."""


def _unpack_ok(call_result: Any, tool_name: str) -> dict:
    """Pull the JSON-decoded content out of a `ConnectorOk` envelope.

    Provider returns `ConnectorOk(content=json.dumps({...}))`; the MCP
    layer wraps that as `{"kind": "ok", "content": "..."}`. We get the
    dict out the other side. Raises _ReauthRequired / RuntimeError on
    non-ok variants so the caller can branch cleanly.
    """
    envelope = getattr(call_result, "structured_content", None)
    if envelope is None:
        envelope = getattr(call_result, "structuredContent", None)
    if not isinstance(envelope, dict):
        raise RuntimeError(
            f"{tool_name}: missing structured envelope (got {type(envelope).__name__})"
        )
    kind = envelope.get("kind")
    if kind == "ok":
        raw = envelope.get("content")
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"{tool_name}: content not JSON ({e})") from e
        if isinstance(raw, dict):
            return raw
        raise RuntimeError(f"{tool_name}: unexpected content type {type(raw).__name__}")
    if kind == "reauth_required":
        raise _ReauthRequired(envelope.get("message") or "reauth required")
    # rate_limited / provider_down / scope_missing / tool_error — all
    # propagate as a generic runtime error; the runner's retry loop will
    # decide whether to retry (provider_down→yes, scope_missing→no, etc).
    msg = envelope.get("message") or f"connector returned kind={kind}"
    raise RuntimeError(f"{tool_name}: {msg}")


def _compute_query_window(last_state: Optional[dict]) -> str:
    """Compute the Gmail query string for the fetch window.

    Bootstrap path (no watermark): `newer_than:1d`.
    Steady state: `newer_than:Nh` where N = ceil(hours since
    last_processed_internal_date), capped at 168 so an extended outage
    doesn't pull a week+ of junk.

    Math is done in pure milliseconds-since-epoch to avoid the
    naive-utcnow().timestamp() trap, which interprets a naive UTC
    datetime as LOCAL time on .timestamp() — off by `local_tz - UTC`
    hours in any non-UTC environment.
    """
    if not last_state:
        return _BOOTSTRAP_WINDOW
    last_ms_raw = last_state.get("last_processed_internal_date")
    if not last_ms_raw:
        return _BOOTSTRAP_WINDOW
    try:
        last_ms = int(last_ms_raw)
    except (TypeError, ValueError):
        return _BOOTSTRAP_WINDOW
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    delta_hours = max(1, (now_ms - last_ms) // 3_600_000 + 1)
    if delta_hours > _STEADY_WINDOW_CAP_HOURS:
        return _BOOTSTRAP_WINDOW
    return f"newer_than:{delta_hours}h"


def _format_emails_for_llm(emails: List[_Email]) -> str:
    """Build the user-message block fed to Haiku. We trim each body to
    keep the prompt manageable — a 50-email briefing with full bodies
    can blow past Haiku's input window."""
    lines: List[str] = []
    for e in emails:
        sender = e.headers.get("From", "(unknown)")
        subject = e.headers.get("Subject", "(no subject)")
        date = e.headers.get("Date", "")
        # Trim body — first ~600 chars is plenty for the LLM to judge
        # priority + extract action items.
        body_excerpt = (e.body or e.snippet or "").strip()
        if len(body_excerpt) > 600:
            body_excerpt = body_excerpt[:600] + "…"
        lines.append(
            f"--- email id={e.id} thread={e.thread_id}\n"
            f"From: {sender}\nSubject: {subject}\nDate: {date}\n\n{body_excerpt}\n"
        )
    return "\n".join(lines)


class EmailBriefingHandler:
    """RoutineHandler implementation for `kind="email_briefing"`."""

    kind = "email_briefing"

    def __init__(self, mcp_client: Any = None, llm_fn: Any = None, writer: Any = None):
        """All three deps are injected so tests can swap mocks without
        monkeypatching modules.

        - `mcp_client`: object with `await call_tool(name, args)`.
          In production, `tool_executor.mcp_client` (a `fastmcp.Client`).
        - `llm_fn`: callable matching `call_anthropic_system` signature.
          Defaults to the real one at first use.
        - `writer`: callable matching `write_routine_message` signature.
          Defaults to the real one at first use.
        """
        self._mcp_client = mcp_client
        self._llm_fn = llm_fn
        self._writer = writer

    async def execute(self, routine: Any, run: Any, db: AsyncSession) -> RoutineResult:
        cfg = routine.config_json or {}
        mcp = self._mcp_client
        if mcp is None:
            return RoutineResult(
                status="failed",
                error_class="no_mcp_client",
                error_detail="RoutineRunner did not provide an MCP client",
            )

        # Ticket 7 (2026-05-13): two semantic modes.
        #
        #   - "since_last_run" (default, legacy): emails that arrived
        #     after the last successful briefing's watermark. Posts
        #     "No new emails since the last briefing" when the inbox
        #     was quiet. This is what the original "morning unread
        #     briefing" target wanted.
        #
        #   - "latest_n": always the most recent N emails regardless
        #     of the watermark. This is what users mean by "give me
        #     the latest 5 every day at 10:58." The bug we're fixing:
        #     before this knob existed, the user's routine landed on
        #     `since_last_run` and after the first run advanced the
        #     watermark, every subsequent run fetched 0 emails (their
        #     inbox was quiet on the inter-run interval). Per-day
        #     repetition of the same 5 emails IS the desired UX here.
        mode = (cfg.get("mode") or "since_last_run").strip()
        if mode not in ("since_last_run", "latest_n"):
            mode = "since_last_run"
        # Mode-aware default: legacy `since_last_run` keeps the 50-email
        # window cap (a real morning briefing can have that many);
        # `latest_n` defaults to 5 (matches the "give me the latest N"
        # ask the bug-sweep surfaced). User can override with
        # `config.max_emails`.
        default_max = 5 if mode == "latest_n" else _MAX_EMAILS_PER_BRIEFING
        try:
            max_emails = int(cfg.get("max_emails") or default_max)
        except (TypeError, ValueError):
            max_emails = default_max
        max_emails = max(1, min(max_emails, _MAX_EMAILS_PER_BRIEFING))

        try:
            emails, fetched_count = await self._fetch(
                mcp, routine.last_state_json,
                mode=mode, max_emails=max_emails,
            )
        except _ReauthRequired as e:
            return RoutineResult(
                status="skipped_reauth",
                error_class="reauth_required",
                error_detail=str(e)[:300],
            )
        except _ToolMissing as e:
            return RoutineResult(
                status="skipped_reauth",
                error_class="tool_missing",
                error_detail=str(e)[:300],
            )
        except Exception as e:
            return RoutineResult(
                status="failed",
                error_class=type(e).__name__,
                error_detail=str(e)[:300],
            )

        if not emails:
            return await self._post_empty(routine, db)

        # Summarize via the agent's active LLM. `call_system_llm` resolves
        # provider + auth via `model_resolver` + `bundle_client` — same
        # path the user's chat uses. Tests inject `self._llm_fn` to skip
        # network; in prod we route through bundle when active.
        llm = self._llm_fn
        if llm is None:
            from app.services.internal_llm import call_system_llm
            llm = call_system_llm

        # Per-routine model with default-cheap policy (2026-05-13 cost
        # audit): when `config_json.model` is unset we default to
        # gpt-4o-mini ($0.15/1M input) instead of falling through to
        # the user's chat-model default. Email briefings are inherently
        # summarization — gpt-4o-mini is excellent at this and the
        # cheapest production-grade option on the platform OpenAI key.
        # Power users wanting "Sonnet for nuance" / "Opus for the
        # morning digest" can still set `config_json.model` explicitly
        # per routine.
        cfg = routine.config_json or {}
        model_choice = (cfg.get("model") or "").strip() or "gpt-4o-mini"

        prompt_body = _format_emails_for_llm(emails)
        if fetched_count > len(emails):
            prompt_body += f"\n(+{fetched_count - len(emails)} more new emails not shown)"

        # Mode-aware system prompt. latest_n must NOT say "no new
        # emails since the last briefing" — by definition the listed
        # emails ARE the latest, not the "new."
        system_prompt = SYSTEM_PROMPT_LATEST_N if mode == "latest_n" else SYSTEM_PROMPT

        summary_text = await llm(
            user_id=routine.user_id,
            operation_type=_OPERATION_TYPE,
            model=model_choice,
            max_tokens=_SUMMARY_MAX_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt_body}],
            timeout=60,
        )
        if not summary_text:
            return RoutineResult(
                status="failed",
                error_class="llm_returned_none",
                error_detail="call_system_llm returned None (timeout / auth / parse)",
            )

        # Post the briefing.
        writer = self._writer
        if writer is None:
            from .message_writer import write_routine_message, broadcast_routine_message
            writer = write_routine_message
            broadcaster = broadcast_routine_message
        else:
            broadcaster = None  # tests inject writer + skip broadcast

        # `model_used` is best-effort metadata for the Message row.
        # The llm_proxy_events log captures the actually-used model
        # and is the source of truth for billing.
        model_used_for_record = model_choice

        msg_id, day_chat_id = await writer(
            db,
            user_id=routine.user_id,
            content=summary_text,
            source=self.kind,
            routine_id=routine.id,
            title=f"Morning briefing — {datetime.utcnow().date().isoformat()}",
            model_used=model_used_for_record,
            # Token counts aren't returned by call_system_llm as a struct;
            # the LLM-proxy log captures them. Leaving null on the Message
            # row is consistent with day_archival.
            tokens_prompt=None,
            tokens_completion=None,
            extra_metadata={"routine_message": True, "routine_id": routine.id,
                            "routine_name": routine.name or "Morning email briefing"},
        )

        # Ticket 2.5 — capture per-channel delivery results so the
        # runner can downgrade outcome to `partial` when any channel
        # skipped silently.
        channel_results: dict[str, dict[str, Any]] = {}
        if broadcaster is not None:
            from .channel_dispatcher import parse_delivery_channels
            broadcast_out = await broadcaster(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=summary_text,
                model_used=model_used_for_record,
                delivery_channels=parse_delivery_channels(routine.config_json),
                routine_name=routine.name or "Morning email briefing",
            )
            # Support both legacy (int) and Ticket-2.5 (dict) return shape.
            if isinstance(broadcast_out, dict):
                channel_results = broadcast_out.get("channel_results", {}) or {}

        new_watermark = self._advance_watermark(routine.last_state_json, emails)
        return RoutineResult(
            status="success",
            emails_fetched=fetched_count,
            summary_message_id=msg_id,
            new_watermark=new_watermark,
            channel_results=channel_results,
            tools_invoked=["gmail__list_messages", "gmail__get_message"],
            metrics={
                "summary_chars": len(summary_text),
                "emails_summarized": len(emails),
                "emails_total_in_window": fetched_count,
            },
        )

    # ------------------------------------------------------------------ fetch
    async def _fetch(
        self,
        mcp_client: Any,
        last_state: Optional[dict],
        *,
        mode: str = "since_last_run",
        max_emails: int = _MAX_EMAILS_PER_BRIEFING,
    ) -> tuple[List[_Email], int]:
        """Return (emails_to_summarize, total_emails_in_window). The
        first list is capped at max_emails; the count tells us if there
        were more we didn't hydrate so the summary can say "+N more".

        `mode="since_last_run"` (default) uses the watermark-driven
        query window. `mode="latest_n"` ignores the watermark — empty
        query + `max_results=max_emails` returns the freshest N
        regardless of when they arrived.

        Each MCP call goes through `async with mcp_client` + the
        `pending_channel="routine"` shim — same contract `tool_executor`
        uses ([tool_executor.py:378-384]). Outside the context, fastmcp's
        Client has no connection and call_tool raises. Channel context
        tells the platform dispatcher which channel issued the call so
        per-channel deny rules (e.g., "no send_message from voice") apply
        correctly.
        """
        # Lazy-import the channel shim — agent_main wires the context var
        # at boot. In test envs the shim degrades to a no-op.
        try:
            from app.agent.mcp_client_auth import set_pending_channel, reset_pending_channel
            ch_token = set_pending_channel("routine")
        except Exception:
            ch_token = None
            reset_pending_channel = None  # type: ignore

        try:
            if mode == "latest_n":
                # No `newer_than` filter — Gmail returns the most-recent
                # N by default-sort. We pass an empty q so Gmail doesn't
                # apply a time gate.
                list_args = {"q": "", "max_results": max_emails}
            else:
                list_args = {
                    "q": _compute_query_window(last_state),
                    "max_results": max_emails,
                }
            async with mcp_client:
                list_result = await mcp_client.call_tool(
                    "gmail__list_messages",
                    list_args,
                )
            list_data = _unpack_ok(list_result, "gmail__list_messages")
            ids = [m["id"] for m in (list_data.get("messages") or []) if m.get("id")]
            total = list_data.get("result_size") or len(ids)

            emails: List[_Email] = []
            for mid in ids[:max_emails]:
                try:
                    async with mcp_client:
                        msg_result = await mcp_client.call_tool(
                            "gmail__get_message", {"message_id": mid}
                        )
                    msg_data = _unpack_ok(msg_result, "gmail__get_message")
                except _ReauthRequired:
                    raise
                except Exception as e:
                    logger.warning(
                        "[email_briefing] get_message id=%s skipped: %s: %s",
                        mid, type(e).__name__, str(e)[:120],
                    )
                    continue
                emails.append(_Email(
                    id=msg_data.get("id", mid),
                    thread_id=msg_data.get("threadId", ""),
                    headers=msg_data.get("headers") or {},
                    snippet=msg_data.get("snippet") or "",
                    body=msg_data.get("body") or "",
                    internal_date=msg_data.get("internalDate"),
                ))
            return emails, total
        finally:
            if ch_token is not None and reset_pending_channel is not None:
                try:
                    reset_pending_channel(ch_token)
                except Exception:
                    pass

    # ------------------------------------------------------------------ empty
    async def _post_empty(self, routine: Any, db: AsyncSession) -> RoutineResult:
        """Write the 'no new emails since the last briefing' message.
        We do post a Message — the user needs the signal that the
        routine ran — but we do NOT advance the watermark (we have no
        new max(internal_date) to advance to)."""
        writer = self._writer
        if writer is None:
            from .message_writer import write_routine_message, broadcast_routine_message
            writer = write_routine_message
            broadcaster = broadcast_routine_message
        else:
            broadcaster = None

        text = "No new emails since the last briefing."
        msg_id, day_chat_id = await writer(
            db,
            user_id=routine.user_id,
            content=text,
            source=self.kind,
            routine_id=routine.id,
            title=f"Morning briefing — {datetime.utcnow().date().isoformat()}",
            model_used=None,
            extra_metadata={"routine_message": True, "routine_id": routine.id,
                            "routine_name": routine.name or "Morning email briefing"},
        )
        channel_results: dict[str, dict[str, Any]] = {}
        if broadcaster is not None:
            from .channel_dispatcher import parse_delivery_channels
            broadcast_out = await broadcaster(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=text,
                model_used=None,
                delivery_channels=parse_delivery_channels(routine.config_json),
                routine_name=routine.name or "Morning email briefing",
            )
            if isinstance(broadcast_out, dict):
                channel_results = broadcast_out.get("channel_results", {}) or {}
        return RoutineResult(
            status="success",
            emails_fetched=0,
            summary_message_id=msg_id,
            new_watermark=None,  # don't regress
            channel_results=channel_results,
            tools_invoked=["gmail__list_messages"],
            metrics={"empty_run": True},
        )

    # ------------------------------------------------------------------ watermark
    @staticmethod
    def _advance_watermark(prev: Optional[dict], emails: List[_Email]) -> Optional[dict]:
        """Pick the max internalDate seen this run. Preserves any other
        fields the watermark dict may carry."""
        dates = [int(e.internal_date) for e in emails if e.internal_date is not None]
        if not dates:
            return prev
        new = dict(prev or {})
        new["last_processed_internal_date"] = max(dates)
        new["last_processed_at"] = datetime.utcnow().isoformat()
        return new
