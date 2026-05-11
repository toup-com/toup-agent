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


# Conservative Haiku cap. 300-word target ≈ 400 tokens; ceiling at 1200
# gives margin without inviting LLM ramble. Day-archival uses 1800 for a
# longer summary; we're tighter.
_HAIKU_MAX_TOKENS = 1200
_HAIKU_MODEL = "claude-haiku-4-5-20251001"

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

        try:
            emails, fetched_count = await self._fetch(mcp, routine.last_state_json)
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

        # Summarize via Haiku.
        llm = self._llm_fn
        if llm is None:
            from app.services.internal_llm import call_anthropic_system
            llm = call_anthropic_system

        prompt_body = _format_emails_for_llm(emails)
        if fetched_count > len(emails):
            prompt_body += f"\n(+{fetched_count - len(emails)} more new emails not shown)"

        summary_text = await llm(
            user_id=routine.user_id,
            operation_type=_OPERATION_TYPE,
            model=_HAIKU_MODEL,
            max_tokens=_HAIKU_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt_body}],
            timeout=60,
        )
        if not summary_text:
            return RoutineResult(
                status="failed",
                error_class="llm_returned_none",
                error_detail="internal_llm.call_anthropic_system returned None (timeout / auth / parse)",
            )

        # Post the briefing.
        writer = self._writer
        if writer is None:
            from .message_writer import write_routine_message, broadcast_routine_message
            writer = write_routine_message
            broadcaster = broadcast_routine_message
        else:
            broadcaster = None  # tests inject writer + skip broadcast

        msg_id, day_chat_id = await writer(
            db,
            user_id=routine.user_id,
            content=summary_text,
            source=self.kind,
            routine_id=routine.id,
            title=f"Morning briefing — {datetime.utcnow().date().isoformat()}",
            model_used=_HAIKU_MODEL,
            # Token counts aren't returned by call_anthropic_system as a struct;
            # the LLM-proxy log captures them. Leaving null on the Message row
            # is consistent with day_archival.
            tokens_prompt=None,
            tokens_completion=None,
        )
        if broadcaster is not None:
            await broadcaster(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=summary_text,
                model_used=_HAIKU_MODEL,
            )

        new_watermark = self._advance_watermark(routine.last_state_json, emails)
        return RoutineResult(
            status="success",
            emails_fetched=fetched_count,
            summary_message_id=msg_id,
            new_watermark=new_watermark,
            metrics={
                "summary_chars": len(summary_text),
                "emails_summarized": len(emails),
                "emails_total_in_window": fetched_count,
            },
        )

    # ------------------------------------------------------------------ fetch
    async def _fetch(self, mcp_client: Any, last_state: Optional[dict]) -> tuple[List[_Email], int]:
        """Return (emails_to_summarize, total_emails_in_window). The
        first list is capped at _MAX_EMAILS_PER_BRIEFING; the count tells
        us if there were more we didn't hydrate so the summary can say
        "+N more".

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
            query = _compute_query_window(last_state)
            async with mcp_client:
                list_result = await mcp_client.call_tool(
                    "gmail__list_messages",
                    {"q": query, "max_results": _MAX_EMAILS_PER_BRIEFING},
                )
            list_data = _unpack_ok(list_result, "gmail__list_messages")
            ids = [m["id"] for m in (list_data.get("messages") or []) if m.get("id")]
            total = list_data.get("result_size") or len(ids)

            emails: List[_Email] = []
            for mid in ids[:_MAX_EMAILS_PER_BRIEFING]:
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
        )
        if broadcaster is not None:
            await broadcaster(
                routine.user_id,
                message_id=msg_id,
                day_chat_id=day_chat_id,
                source=self.kind,
                content=text,
                model_used=None,
            )
        return RoutineResult(
            status="success",
            emails_fetched=0,
            summary_message_id=msg_id,
            new_watermark=None,  # don't regress
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
