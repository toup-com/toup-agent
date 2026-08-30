"""
T1e — Connector dispatcher (the runtime hot path).

Every tool call from every agent flows through `dispatcher.execute(...)`.
Single entry point; everything else is internal.

What this module owns:

  - Pre-flight checks: manifest tool lookup, channel-policy resolution
    (manifest deny list + `mutates: true → voice/telegram default-deny`
    per architecture §4.4).
  - Vault read; routing of None/`reauth_required` to ConnectorReauthRequired.
  - Lazy refresh on tool call (architecture §3.4) with per-identity
    asyncio.Lock so 50 concurrent calls coalesce into ONE refresh.
  - Audit-then-act (§3.6): EVENT_TOOL_CALLED is committed BEFORE
    `provider.execute()` runs. If audit commit fails, the provider
    call never happens. EVENT_TOOL_SUCCEEDED / EVENT_TOOL_FAILED is
    committed after.
  - Output redaction: per-tool `output_redaction` list strips named
    fields from BOTH input and output metadata before they hit the
    audit row. The result returned to the caller is unredacted —
    redaction is for the durable audit, not for the LLM.

What this module deliberately does NOT do:

  - HTTP / FastAPI / MCP transport (T1f registers these tools on the
    platform MCP server; T1f translates dispatcher results to MCP
    response shape).
  - Per-user channel-policy overrides from `ConnectorUserPreference`
    (T2c wires the UI; the dispatcher gets a hook then).
  - Rate-limit ENFORCEMENT — `manifest_tool.rate_limit` is informational
    only in v1. TODO marker below; T5a (observability) or a follow-up
    will own real rate-limit logic.
  - Metrics — Prometheus counters land in T5a. INFO logs are the
    operational signal until then.
  - Tools-cache invalidation — that's T1h's `/api/agent/refresh-tools`.

Result types are the sum-type from `app/connectors/base.py` (architecture
§2.5). The dispatcher returns them as-is; T1f does the LLM-tool-result
serialisation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.connectors.base import (
    ConnectorConfirmationRequired,
    ConnectorContext,
    ConnectorOk,
    ConnectorProviderDown,
    ConnectorRateLimited,
    ConnectorReauthRequired,
    ConnectorResult,
    ConnectorScopeMissing,
    ConnectorToolError,
    RefreshFailed,
)
from app.services import connector_vault as vault
from app.services.connector_registry import get_registry
from app.services.connector_vault import (
    VaultAuditError,
    # Cross-module import of the audit primitive is intentional — the
    # dispatcher and the vault both implement audit-then-act and
    # duplicating the helper would invite drift. Underscore is
    # convention not enforcement; treating this as a shared internal
    # primitive is the lesser smell.
    _audit_then_commit,
)
from app.db.models import (
    EVENT_REFRESH_FAILED,
    EVENT_REFRESH_SUCCEEDED,
    EVENT_TOOL_CALLED,
    EVENT_TOOL_ELEVATION_REQUIRED,
    EVENT_TOOL_FAILED,
    EVENT_TOOL_SUCCEEDED,
)

logger = logging.getLogger(__name__)


# ─── Tunables ─────────────────────────────────────────────────────────


# Channels that can render a confirmation card and collect a tap. An
# `elevation: true` tool called from anywhere else is REFUSED, never
# silently executed — see the gate in `execute`. Kept as a frozenset
# next to the other channel policy so the two are read together.
#
# `automation_thread` joined in R38, and only once the surface existed:
# `thread_agent` renders each staged action as a `needs_you` turn with
# `fix="approve"` carrying its `pending_action_id`, and the approve
# endpoint re-enters `execute` with `approved_action_id` set — which
# `_resolve_channel_policy` now honours, so the tap actually runs the
# call. Adding the channel here without that would have been a card
# nobody could draw; adding the card without the policy change would
# have been a button that reports failure on every tap.
_CONFIRMABLE_CHANNELS = frozenset({"web", "app", "mobile",
                                   "automation_thread"})

# Channels where a MUTATING connector tool is staged for the user's
# approval rather than denied. `automation_thread` is attended — the
# user is sitting in the thread, watching — so the honest answer to
# "post that to Slack", asked inside the automation whose job is
# posting to Slack, is a card and not a refusal. Every write from here
# meets the elevation gate whether or not the manifest marks the tool
# `elevation: true`, which is the same posture a confirm-mode
# automation grant already has.
_MUTATES_CONFIRM_CHANNELS = frozenset({"automation_thread"})


def stages_writes_for_approval(channel: str) -> bool:
    """Whether a mutating call on `channel` is staged for a tap rather
    than refused — and, therefore, whether an approval offered on that
    channel can actually be honoured.

    Exported so the surface that DRAWS the approve button and the code
    that would EXECUTE it read one predicate. `thread_agent` asks
    before offering approval; if this ever answered False the turn says
    it cannot run the call instead of offering a tap that fails.
    """
    return (channel or "") in _MUTATES_CONFIRM_CHANNELS

# How long a staged draft stays actionable. Long enough to survive
# "I'll deal with this after lunch", short enough that a card found by
# scrolling through last week cannot fire.
_PENDING_ACTION_TTL_HOURS = 24

# Refresh skew: refresh if access token expires within this many seconds.
# Same value as agent_key_rotation_verify; the underlying problem is the
# same — don't hand the LLM a token that's about to die mid-call.
_REFRESH_SKEW_SECONDS = 30

# Voice and telegram are the channels where mutating tool calls have no
# user-confirmation surface (no chip UI, no inline approve gate). Default-
# deny mutating tools on those channels; per-tool channel_policy.deny in
# the manifest is additive on top.
#
# "autopilot" (Autopilot arc PR7): autonomous mission ticks run with
# NOBODY watching — outward mutation (gmail send, calendar write, …)
# is default-denied. A user's explicit per-tool allow
# (ConnectorUserPreference) still overrides, same resolution order as
# voice/telegram.
_MUTATES_DEFAULT_DENY_CHANNELS = frozenset({"voice", "telegram", "autopilot"})

# No-user-present channels: background routines / triggers run a fully
# tool-enabled agent turn with nobody watching, so injected content in an
# ingested email/web/doc can drive a mutating connector call unattended
# (docs/security/audit-2026.md INJ-1, the Critical). When injection_fencing_v2
# is on we default-deny mutating connectors on these channels too. Flag-gated
# because it changes unattended-automation behaviour (an agent_task routine
# can no longer auto-send email until a confirmation path exists — INJ-4/G6).
# "autopilot" is already in the default-deny set above; this is the additive
# rest of the unattended surface.
_MUTATES_UNATTENDED_DENY_CHANNELS = frozenset({
    "routine", "trigger", "heartbeat", "cron", "agent_task",
    "email_briefing", "background",
    # A spawned sub-agent / app-builder turn runs autonomously with nobody
    # watching each tool call, so it is an unattended surface too. Without
    # these, a denied unattended parent (routine/cron/trigger) could LAUNDER a
    # mutating connector call through a child sub-agent that hard-codes
    # channel="subagent" (audit-2026 re-audit round 7, INJ-1 follow-up). A
    # user's explicit per-tool ConnectorUserPreference allow still overrides,
    # same resolution order as autopilot.
    "subagent", "app_builder",
    # The automation thread's agent turn (R33). The user IS present, and
    # since R38 it also has a surface to approve on — so it is in
    # `_MUTATES_CONFIRM_CHANNELS` below, which SUBTRACTS it from this set
    # inside `_resolve_channel_policy`. It stays listed here on purpose:
    # the subtraction is what carries the reasoning, and deleting the name
    # would make "why is this channel allowed to write?" unanswerable from
    # the policy the way it was before the surface existed.
    "automation_thread",
})


# ─── Per-identity refresh-coalescing locks ────────────────────────────


# `dict[identity_id, asyncio.Lock]`. Locks are cheap and bounded by
# (active users × connectors). v1 doesn't need cleanup; if production
# memory ever shows this growing unbounded, an LRU wrapper or weakref
# scheme is the easy fix. Keep it simple.
_refresh_locks: dict[str, asyncio.Lock] = {}


def _lock_for(identity_id: str) -> asyncio.Lock:
    lock = _refresh_locks.get(identity_id)
    if lock is None:
        lock = asyncio.Lock()
        _refresh_locks[identity_id] = lock
    return lock


# ─── Automation grant gate (Round 26) ────────────────────────────────


async def _resolve_automation_grant(
    db: AsyncSession,
    *,
    user_id: str,
    connector_id: str,
    entry,
    manifest_tool,
    tool_input: dict,
    grant_id: Optional[str],
    approved_action_id: Optional[str],
):
    """Verify a mutating automation call against its standing grant.

    Returns the AutomationGrant row when the call may proceed, None for
    non-mutating calls (or a per-call human approval), or a
    ConnectorToolError refusing the call. Every refusal is fail-closed
    and names the reason — an automation that cannot write must say so
    in its run ledger, not half-succeed.

    Side effect on success: the grant's cadence counters are
    incremented and committed BEFORE the provider runs. Conservative on
    purpose — a failed provider call still consumed budget, which is
    the safe direction for an unattended writer.
    """
    from sqlalchemy import select as _select
    from app.db.models import AutomationGrant

    if not manifest_tool.mutates:
        return None
    if approved_action_id is not None:
        # The human approved exactly this call via the guarded-UPDATE
        # claim in connector_pending_actions — the strongest per-call
        # authorization there is. The standing grant (if any) is not
        # consulted and its cadence budget is not charged.
        return None
    if not grant_id:
        return ConnectorToolError(
            message=(
                f"{manifest_tool.name!r} modifies data and no approved "
                f"permission backs this automation call. Automations "
                f"fail closed — ask the user for permission first."
            ),
            retryable=False,
        )

    row = (await db.execute(
        _select(AutomationGrant)
        .where(AutomationGrant.id == grant_id)
        .where(AutomationGrant.user_id == user_id)
    )).scalar_one_or_none()
    if row is None:
        return ConnectorToolError(
            message="The permission backing this automation no longer "
                    "exists. Nothing was sent.",
            retryable=False,
        )
    if row.status != "approved":
        return ConnectorToolError(
            message=f"The permission backing this automation is "
                    f"{row.status!r}, not approved. Nothing was sent.",
            retryable=False,
        )
    if row.connector_id != connector_id or row.tool_name != manifest_tool.name:
        return ConnectorToolError(
            message="The approved permission covers a different action "
                    "than this call. Nothing was sent.",
            retryable=False,
        )

    # Pinned-target check. The param name comes from the manifest's
    # automation block; a write tool WITHOUT a declared target param
    # cannot be grant-gated and is refused (the registry lint enforces
    # the declaration, this is the runtime backstop).
    auto_block = getattr(entry.manifest, "automation", None)
    target_param = (
        auto_block.target_param_by_action.get(manifest_tool.name)
        if auto_block is not None else None
    )
    if not target_param:
        return ConnectorToolError(
            message=f"{manifest_tool.name!r} declares no pinned-target "
                    f"parameter, so no grant can cover it. Nothing was sent.",
            retryable=False,
        )
    try:
        target = json.loads(row.target_json or "{}")
    except (ValueError, TypeError):
        target = {}
    actual = str(tool_input.get(target_param) or "").strip()
    pinned = str(target.get("id") or "").strip()
    if not actual or not pinned or actual != pinned:
        return ConnectorToolError(
            message=(
                f"This automation may only write to "
                f"{target.get('label') or pinned or 'its approved target'} "
                f"— the call targeted {actual or 'nothing'} instead. "
                f"Nothing was sent."
            ),
            retryable=False,
        )

    # Cadence budget. Counters are day/hour-keyed on the row itself; a
    # new period resets them in the same UPDATE that increments.
    try:
        cadence = json.loads(row.cadence_json) if row.cadence_json else {}
    except (ValueError, TypeError):
        cadence = {}
    now = datetime.utcnow()
    day_key = now.strftime("%Y-%m-%d")
    hour_key = now.strftime("%Y-%m-%dT%H")
    if row.uses_day_key != day_key:
        row.uses_day_key = day_key
        row.uses_today = 0
    if row.uses_hour_key != hour_key:
        row.uses_hour_key = hour_key
        row.uses_this_hour = 0
    per_day = cadence.get("per_day")
    per_hour = cadence.get("per_hour")
    if per_day is not None and row.uses_today >= int(per_day):
        await db.rollback()
        return ConnectorToolError(
            message=f"This automation's daily budget ({per_day} writes) "
                    f"is used up — it resets at midnight UTC.",
            retryable=True,
        )
    if per_hour is not None and row.uses_this_hour >= int(per_hour):
        await db.rollback()
        return ConnectorToolError(
            message=f"This automation's hourly budget ({per_hour} writes) "
                    f"is used up — try again next hour.",
            retryable=True,
        )
    row.uses_today += 1
    row.uses_this_hour += 1
    row.last_used_at = now
    await db.commit()
    return row


# ─── Public refresh helpers (shared with the health-probe scheduler) ─
#
# The health probe needs the SAME refresh-on-expiring logic the
# dispatcher runs on tool calls — otherwise active identities flip
# to provider_down 1 h after the user's last action (Google access
# tokens expire). Re-using the dispatcher's helpers (rather than
# duplicating) keeps the per-identity coalescing lock global so a
# tool call and a probe can't both call provider.refresh() at the
# same instant.
#
# Definitions live below at `_needs_refresh` / `_refresh_with_coalescing`;
# these aliases are bound at the bottom of the module so callers can
# import them by their public names.


# ─── Public entry point ──────────────────────────────────────────────


async def execute(
    db: AsyncSession,
    user_id: str,
    connector_id: str,
    tool_name: str,
    tool_input: dict,
    channel: str = "web",
    *,
    agent_request_id: Optional[str] = None,
    approved_action_id: Optional[str] = None,
    grant_id: Optional[str] = None,
    automation_id: Optional[str] = None,
    exclude_metering: bool = False,
) -> ConnectorResult:
    """Run one connector tool call end-to-end.

    Returns a `ConnectorResult` subclass — never raises for normal
    error paths. Unhandled exceptions surface as `ConnectorToolError`
    so the LLM gets a well-shaped message instead of a 500.

    `approved_action_id` lifts the `elevation: true` confirmation gate
    for exactly this call. ONLY `connector_pending_actions.py` passes
    it, and only after it has atomically claimed the row (flipped
    `pending → approved` in a single guarded UPDATE). It is keyword-only
    and the MCP tool handler never supplies it, so nothing on the
    agent-facing path can approve its own send.

    `grant_id` (Round 26) names an approved standing AutomationGrant.
    Only valid with `channel="automation"`, and only the automations
    RPC endpoint passes it — the MCP handler never does. The gate at
    step 1.7 verifies status, ownership, action, pinned target and
    cadence budget against the row, failing closed on any mismatch; a
    mutating call on the automation channel WITHOUT a verifiable grant
    is refused outright.

    `automation_id` (Round 30) is metering attribution only: the
    automations RPC passes it so the charge's ledger row names the
    automation that spent the credit. It gates nothing — the grant
    machinery above is the enforcement.
    """
    started = time.monotonic()
    # Phase-level timing breakdown. Logged at the end of `execute`
    # alongside the outcome so ops can spot regressions (DB write
    # latency spike, Google API slow path, etc.) at a glance. Each
    # phase records cumulative ms since `started`.
    phases: dict[str, float] = {}

    def _mark(phase: str) -> None:
        phases[phase] = round((time.monotonic() - started) * 1000.0, 1)

    user_hash = _hash_user(user_id)

    # 1. Manifest tool lookup.
    entry = get_registry().get(connector_id)
    if entry is None:
        _log(user_hash, connector_id, tool_name, channel, "unknown_connector", started)
        return ConnectorToolError(
            message=f"unknown connector {connector_id!r}",
            retryable=False,
        )
    manifest_tool = next(
        (t for t in entry.manifest.tools if t.name == tool_name), None,
    )
    if manifest_tool is None:
        _log(user_hash, connector_id, tool_name, channel, "unknown_tool", started)
        return ConnectorToolError(
            message=f"connector {connector_id!r} does not declare tool {tool_name!r}",
            retryable=False,
        )

    # 1.5 T5b — operator force-quarantine. Highest priority: blocks
    #     EVERYONE on this connector regardless of vault state, user
    #     prefs, or channel. Surface the operator's reason verbatim
    #     so the user knows reconnecting won't help.
    from app.services import connector_quarantine as _q
    q_entry = _q.is_quarantined(connector_id)
    if q_entry is not None:
        _log(user_hash, connector_id, tool_name, channel, "quarantined", started)
        return ConnectorToolError(
            message=(
                f"Service paused by operator: {q_entry.reason[:200]}. "
                f"Reconnection won't help — the operator must lift the "
                f"quarantine first. Try again later."
            ),
            retryable=True,  # eligible for retry once lifted
        )

    # 1.7 Automation-channel gate (Round 26). The automation channel is
    #     machine-driven with nobody watching, so it does not ride the
    #     generic channel-policy layer: reads pass the normal gates
    #     below, and a MUTATING call is legal only when backed by an
    #     approved standing grant whose pinned target matches the
    #     actual arguments and whose cadence budget has room. Fail
    #     closed on every mismatch. `approved_action_id` (a per-call
    #     human approval, claimed by guarded UPDATE) supersedes the
    #     standing-grant requirement for exactly that call.
    automation_grant = None
    if channel == "automation":
        gate = await _resolve_automation_grant(
            db,
            user_id=user_id,
            connector_id=connector_id,
            entry=entry,
            manifest_tool=manifest_tool,
            tool_input=tool_input,
            grant_id=grant_id,
            approved_action_id=approved_action_id,
        )
        if isinstance(gate, ConnectorToolError):
            _log(user_hash, connector_id, tool_name, channel,
                 "automation_grant_denied", started)
            return gate
        automation_grant = gate
    elif grant_id is not None:
        # A grant ref outside the automation channel is a programming
        # error somewhere above us — refuse rather than half-honor it.
        _log(user_hash, connector_id, tool_name, channel,
             "grant_wrong_channel", started)
        return ConnectorToolError(
            message="grant_id is only valid on the automation channel",
            retryable=False,
        )

    # 2 + 3. Channel-policy resolution + vault read in PARALLEL.
    #
    # Pre-2026-05-11 these two DB queries ran serially: preference
    # SELECT (~50-150ms on Railway+pgbouncer), then vault SELECT
    # (~50-150ms), adding up to ~100-300ms before the provider call
    # even started. They're independent — the channel-policy
    # resolution only needs preference rows, the vault.get only
    # needs the identity row — so we fan them out with asyncio.gather.
    # Saves one full DB round-trip per tool call. No security
    # impact: both results are checked below before provider.execute
    # is invoked, and the audit-then-act gate (step 5) is the
    # security-relevant boundary, not the order of pre-flight reads.
    #
    # asyncpg refuses two queries concurrently on the same connection,
    # so the parallel vault read uses a *separate* session — opened
    # for the read, closed when the gather returns. The dispatcher's
    # caller-supplied `db` session is reserved for the audit-then-act
    # commit later in this function.
    #
    # Resolution order (architecture §4.4) for the preference branch:
    #   1. ConnectorUserPreference.enabled = false → reject (kill switch)
    #   2. ConnectorUserPreference.per_tool_overrides[tool][channel] = false → reject
    #   3. ConnectorUserPreference.per_tool_overrides[tool][channel] = true → ALLOW
    #      (skip manifest layer entirely — explicit user grant overrides
    #       the mutating-tool default-deny on voice/telegram)
    #   4. Manifest channel_policy.deny → reject
    #   5. Manifest mutates: true AND channel in {voice, telegram} → reject
    #   6. Default
    from app.db.database import async_session_maker as _make_session

    async def _vault_get_with_own_session():
        async with _make_session() as db2:
            return await vault.get(db2, user_id, connector_id)

    # Defensive fallback: if the parallel-gather path raises for any
    # reason (asyncpg connection acquisition under contention, session
    # construction error, whatever), fall back to the serial path
    # using the caller's `db`. Slower (~50-150ms extra on pgbouncer)
    # but keeps every connector tool working. NEVER swallow without
    # logging — the alert tells us a known-good fast path stopped
    # working.
    try:
        pref_decision, identity = await asyncio.gather(
            _resolve_user_preference(db, user_id, connector_id, tool_name, channel),
            _vault_get_with_own_session(),
        )
    except Exception as e:
        logger.warning(
            "[dispatcher] parallel pre-flight raised (%s: %s) — "
            "falling back to serial for this call. Investigate "
            "before this pattern goes wide.",
            type(e).__name__, e,
        )
        pref_decision = await _resolve_user_preference(
            db, user_id, connector_id, tool_name, channel,
        )
        identity = await vault.get(db, user_id, connector_id)
    _mark("preflight")
    if isinstance(pref_decision, ConnectorToolError):
        _log(user_hash, connector_id, tool_name, channel, "pref_denied", started)
        return pref_decision
    if pref_decision is _PREF_EXPLICIT_ALLOW:
        # User explicitly enabled this tool×channel — bypass manifest
        # checks entirely. The dispatcher still enforces vault status
        # and quarantine; it's only the channel-policy layer that's
        # skipped.
        pass
    else:
        # pref_decision is None — fall through to manifest defaults.
        channel_check = _resolve_channel_policy(manifest_tool, channel)
        if channel_check is not None:
            _log(user_hash, connector_id, tool_name, channel, "channel_denied", started)
            return channel_check

    # 3.0 Vault read result (fetched in parallel above).
    if identity is None:
        _log(user_hash, connector_id, tool_name, channel, "no_identity", started)
        return ConnectorReauthRequired(
            reauth_url=f"/agent/integrations/{connector_id}",
        )
    if identity.status != "active":
        _log(user_hash, connector_id, tool_name, channel, f"status_{identity.status}", started)
        return ConnectorReauthRequired(
            reauth_url=f"/agent/integrations/{connector_id}",
        )

    # 3.5 Per-identity read-only gate. The MCP tool filter already
    #     drops mutating tools from the agent's tool list when the
    #     identity is read-only — this is defense-in-depth for any
    #     code path that bypassed the filter (cached tool spec, raw
    #     dispatcher call from a test, etc).
    if getattr(identity, "read_only", False) and manifest_tool.mutates:
        _log(user_hash, connector_id, tool_name, channel, "read_only_blocked", started)
        return ConnectorToolError(
            message=(
                f"{connector_id} is currently set to read-only — "
                f"tool {tool_name!r} would modify data and is blocked. "
                f"Toggle off read-only on the integrations page to enable."
            ),
            retryable=False,
        )

    # 3.7 Elevation gate. `elevation: true` in the manifest means this
    #     call does something the user must see and approve BEFORE it
    #     happens — sending mail, posting publicly, writing a calendar
    #     event. We stage the arguments and return without touching the
    #     provider; the user reviews (and may edit) them on a card in
    #     chat, and `POST /api/connectors/pending-actions/{id}/approve`
    #     re-enters this function with `approved_action_id` set.
    #
    #     Deliberately placed AFTER the identity/read-only checks — a
    #     disconnected or read-only connector should say so rather than
    #     stage a draft that could never run — and BEFORE the refresh,
    #     the EVENT_TOOL_CALLED audit, and the credit pre-flight, none
    #     of which should fire for a call that is not being made.
    #
    #     `approved_action_id` is keyword-only and the MCP handler never
    #     passes it, so the agent-facing path cannot lift its own gate.
    _needs_card = manifest_tool.elevation and approved_action_id is None
    if _needs_card and automation_grant is not None and automation_grant.mode == "auto":
        # Round 26: an approved auto-mode grant IS the user's standing
        # approval for this exact (tool, pinned target) — the per-call
        # card would ask a question they already answered.
        _needs_card = False
    if (
        not _needs_card
        and automation_grant is not None
        and automation_grant.mode == "confirm"
        and approved_action_id is None
    ):
        # Round 26: a confirm-mode grant previews EVERY fire, even for
        # write tools the manifest does not mark elevation (drafts).
        _needs_card = True
    if (
        approved_action_id is None
        and manifest_tool.mutates
        and channel in _MUTATES_CONFIRM_CHANNELS
    ):
        # R38. On a staging channel EVERY mutating tool draws a card,
        # not just the `elevation: true` ones — otherwise a `mutates`
        # tool the manifest does not elevate (a draft, a label change)
        # would run unattended on a surface whose whole licence to write
        # is that the user approves each call.
        #
        # LAST, and unconditionally, so nothing above can clear it. An
        # auto-mode automation grant is standing approval for the
        # AUTOMATION's own scheduled write, not for a call the agent
        # made in a thread; today `grant_id` is refused outside the
        # `automation` channel so that branch cannot fire here at all,
        # and this ordering is what keeps that true if it ever can.
        _needs_card = True
    if _needs_card:
        if channel not in _CONFIRMABLE_CHANNELS and automation_grant is None:
            # Fail SAFE. Every elevation:true tool today is also
            # `mutates: true`, so the channel policy above has already
            # denied these on voice/telegram/unattended — this is the
            # backstop for a future elevated-but-not-mutating tool, or a
            # new channel added without revisiting this list. Silently
            # executing because we have nowhere to draw a card is the
            # one outcome that must never happen. (A confirm-mode
            # automation grant is exempt: its card lands in the day
            # chat + pending list, both real surfaces.)
            _log(user_hash, connector_id, tool_name, channel,
                 "elevation_unconfirmable_channel", started)
            return ConnectorToolError(
                message=(
                    f"{tool_name!r} needs your confirmation before it runs, "
                    f"and the {channel!r} channel cannot show a confirmation "
                    f"card. Ask again from the Toup app or the web chat."
                ),
                retryable=False,
            )
        staged = await _stage_pending_action(
            db,
            user_id=user_id,
            connector_id=connector_id,
            tool_name=tool_name,
            tool_input=tool_input,
            channel=channel,
            agent_request_id=agent_request_id,
            manifest_tool=manifest_tool,
        )
        _mark("elevation_staged")
        _log(user_hash, connector_id, tool_name, channel, "elevation_staged", started)
        return staged

    # 4. Refresh-on-expiring (lazy, coalesced via per-identity lock).
    if _needs_refresh(identity):
        identity, refresh_outcome = await _refresh_with_coalescing(
            db, entry, identity,
        )
        if isinstance(refresh_outcome, ConnectorReauthRequired):
            _log(user_hash, connector_id, tool_name, channel, "refresh_failed", started)
            return refresh_outcome

    # 5. Audit-then-act. EVENT_TOOL_CALLED is the durable point — if
    #    this commit fails, provider.execute() never runs.
    try:
        await _audit_then_commit(
            db,
            user_id=user_id,
            connector_id=connector_id,
            event_type=EVENT_TOOL_CALLED,
            channel=channel,
            tool_name=tool_name,
            agent_request_id=agent_request_id,
            metadata={"input": _redact(tool_input, manifest_tool.output_redaction)},
        )
        _mark("audit_in")
    except VaultAuditError as e:
        # Fail-closed: provider was NOT called.
        _log(user_hash, connector_id, tool_name, channel, "audit_failed", started)
        logger.error(
            "[dispatcher] audit-then-act gate failed for user=%s tool=%s: %s",
            user_hash, tool_name, e,
        )
        return ConnectorToolError(
            message="audit log unavailable — refusing to invoke provider",
            retryable=True,
        )

    # 5.5 Tool-input safety net: `gmail__list_messages` and
    #     `outlook__list_messages` BOTH manifest `include_body: true` as
    #     the default, but some LLMs (gpt-5.5 in particular) write the
    #     explicit `include_body: false` into tool_input because they
    #     treat the false-shape as the conservative pick. That fights
    #     us — the slow list→get pattern (4 calls × ~10 s) is exactly
    #     what include_body=true was built to skip. Force the default
    #     here when the LLM omitted the parameter; we still honour an
    #     explicit `false` for the rare bulk-enumeration ask.
    if tool_name in ("gmail__list_messages", "outlook__list_messages"):
        if "include_body" not in tool_input:
            tool_input = {**tool_input, "include_body": True}

    # 5.7 Credit pre-flight — integration-credits bucket. Run before
    # the provider call so a credit-poor tenant gets a clean
    # ConnectorToolError instead of a successful tool call they can't
    # pay for. Shadow-mode (credit_enforcement_enabled=False) is a
    # no-op here; deduction still happens post-success below.
    #
    # `exclude_metering` (Round 26): the automations e2e harness runs
    # real dispatches that must not land in the ledger. The RPC only
    # honors mode='e2e' outside production, so this cannot be a
    # billing bypass for a live tenant.
    if exclude_metering:
        # flat_fee=None also disarms the 7.5 post-success charge.
        flat_fee = None
    else:
        try:
            from app.services.credit_service import (
                credit_service as _credit,
                _flat_fee_for_tool,
                BUCKET_INTEGRATION,
            )
            from app.config import settings as _settings
            flat_fee = _flat_fee_for_tool(tool_name, tool_input)
            if getattr(_settings, "credit_enforcement_enabled", False):
                pre = await _credit.check_balance(
                    db, user_id, BUCKET_INTEGRATION, flat_fee,
                )
                if not pre.success:
                    _log(user_hash, connector_id, tool_name, channel,
                         "credits_insufficient", started)
                    return ConnectorToolError(
                        message=(
                            "You're out of integration credits for this month. "
                            "Upgrade your plan or wait for the next renewal to "
                            "use connector tools again."
                        ),
                        retryable=False,
                    )
        except Exception as _credit_pre_err:
            logger.warning(
                "[credits] connector pre-flight failed user=%s connector=%s tool=%s: %s",
                user_hash[:8], connector_id, tool_name, _credit_pre_err,
            )
            flat_fee = None

    # 6. Provider call. Catches both raised exceptions AND the
    #    ConnectorResult sum-type (which is the normal return shape).
    ctx = ConnectorContext(
        user_id=user_id,
        channel=channel,
        request_id=agent_request_id or "no-id",
        # Hand the provider the already-decrypted access token so it
        # doesn't have to re-read the vault. Saves one DB round-trip
        # per tool call (~100-300 ms on Railway+pgbouncer).
        access_token=identity.access_token,
    )
    try:
        result = await entry.provider.execute(tool_name, tool_input, ctx)
        _mark("provider")
    except Exception as e:
        # Unexpected — providers are expected to return sum-type
        # variants for known errors. Map to ConnectorProviderDown when
        # we can't tell, with a short message; full exception goes to
        # the audit row metadata.
        logger.exception(
            "[dispatcher] provider.execute raised unexpectedly for "
            "user=%s connector=%s tool=%s",
            user_hash, connector_id, tool_name,
        )
        await _audit_outcome(
            db,
            user_id=user_id,
            connector_id=connector_id,
            tool_name=tool_name,
            channel=channel,
            agent_request_id=agent_request_id,
            success=False,
            metadata={
                "exception_class": type(e).__name__,
                "exception_str": str(e)[:300],
            },
        )
        _log(user_hash, connector_id, tool_name, channel, "provider_exception", started)
        return ConnectorProviderDown(provider_status_url=None)

    # 7. Audit success/failure based on result variant.
    success = isinstance(result, ConnectorOk)
    await _audit_outcome(
        db,
        user_id=user_id,
        connector_id=connector_id,
        tool_name=tool_name,
        channel=channel,
        agent_request_id=agent_request_id,
        success=success,
        metadata=_outcome_metadata(result, manifest_tool.output_redaction),
    )
    _mark("audit_out")

    # 7.5 Credit deduction (integration bucket, flat-fee). Only on
    # ConnectorOk — rate-limited / reauth-required / provider-down all
    # return early without charging the user. Idempotency key combines
    # connector+tool+agent_request_id so an SDK-level retry of the
    # exact same tool call doesn't double-charge.
    if success and flat_fee is not None:
        try:
            from app.services.credit_service import (
                credit_service as _credit,
                BUCKET_INTEGRATION,
            )
            from app.db.models import LEDGER_TOOL_CALL
            idemp = f"connector:{connector_id}:{tool_name}:{agent_request_id or 'no-id'}"
            await _credit.try_charge(
                db, user_id, LEDGER_TOOL_CALL, BUCKET_INTEGRATION, flat_fee,
                idempotency_key=idemp, event_id=agent_request_id,
                metadata={
                    "connector_id": connector_id,
                    "tool_name": tool_name,
                    "channel": channel,
                    # R30 credit-metering defect: name the automation
                    # that spent the credit (same event_type/bucket as
                    # every chat tool call, so dashboards aggregate
                    # identically; this key is the only difference).
                    **({"automation_id": automation_id}
                       if automation_id else {}),
                },
            )
            # try_charge only FLUSHES — the audit above committed via
            # _audit_then_commit, but nothing committed the charge, and
            # both real callers (the connector MCP handler and the
            # automations dispatch RPC) close their session without
            # committing, so the ledger row was rolled back at close.
            # That is the R30 "dispatches write ZERO credit_ledger
            # rows" defect. The dispatcher owns its transaction
            # boundaries (audit-then-act already commits mid-flight),
            # so the charge commits here too.
            await db.commit()
        except Exception as _credit_charge_err:
            logger.warning(
                "[credits] connector post-charge failed user=%s connector=%s tool=%s: %s",
                user_hash[:8], connector_id, tool_name, _credit_charge_err,
            )

    # 8. Log + return. Result variants other than Ok are normal results,
    #    not errors at the log level — but rate-limit and provider-down
    #    deserve WARN so ops can grep for sustained patterns.
    outcome_label = type(result).__name__
    # Phase-timing breakdown — emitted with every dispatch so a
    # production tail latency spike can be attributed to a phase
    # (DB pre-flight vs provider call vs audit write) without
    # re-running with extra instrumentation. Tiny per-line, formatted
    # as `phase=Xms` pairs so log search tools can grep them.
    phase_str = " ".join(f"{k}={v}ms" for k, v in phases.items())
    if isinstance(result, (ConnectorRateLimited, ConnectorProviderDown)):
        _log(
            user_hash, connector_id, tool_name, channel,
            outcome_label, started, level="WARNING", phases=phase_str,
        )
    else:
        _log(
            user_hash, connector_id, tool_name, channel,
            outcome_label, started, phases=phase_str,
        )
    # T5a — emit Prometheus counters/histogram. Bounded labels: connector
    # id and tool name come from the manifest registry, channel is one
    # of five values, outcome is one of seven sum-type variants.
    try:
        from app.services import connector_metrics as _m
        elapsed_ms = (time.monotonic() - started) * 1000
        _m.inc(_m.M_TOOL_CALLS, labels={
            "connector": connector_id,
            "tool": tool_name,
            "channel": channel,
            "outcome": outcome_label,
        })
        _m.observe_ms(_m.M_DISPATCH_LATENCY, elapsed_ms, labels={
            "connector": connector_id,
            "tool": tool_name,
            "channel": channel,
        })
    except Exception:
        # Metrics failure must NEVER affect tool dispatch.
        pass
    return result


# ─── Channel policy ──────────────────────────────────────────────────


# Sentinel for the "user explicitly granted this tool×channel"
# decision. Identity-comparable via `is`. Distinct from None (which
# means "preference not consulted / no opinion") and from
# ConnectorToolError (which means "preference explicitly denies").
class _PrefAllowSentinel:
    __slots__ = ()


_PREF_EXPLICIT_ALLOW = _PrefAllowSentinel()


async def _resolve_user_preference(
    db: AsyncSession,
    user_id: str,
    connector_id: str,
    tool_name: str,
    channel: str,
) -> Any:
    """T2c — per-user override layer.

    Three return shapes:
      - `None`                      → no preference / no opinion;
                                      caller falls through to manifest
                                      layer.
      - `ConnectorToolError(...)`   → preference explicitly denies;
                                      caller returns it as the
                                      dispatch result.
      - `_PREF_EXPLICIT_ALLOW`      → preference explicitly grants
                                      this tool×channel; caller
                                      SKIPS the manifest layer
                                      entirely (overrides
                                      mutating-tool default-deny on
                                      voice/telegram).
    """
    from sqlalchemy import select as _select
    from app.db.models import ConnectorUserPreference
    pref = (
        await db.execute(
            _select(ConnectorUserPreference)
            .where(ConnectorUserPreference.user_id == user_id)
            .where(ConnectorUserPreference.connector_id == connector_id)
        )
    ).scalar_one_or_none()
    if pref is None:
        return None

    if not pref.enabled:
        return ConnectorToolError(
            message=(
                f"Connector {connector_id!r} is disabled for this user. "
                f"Re-enable it from /agent/integrations."
            ),
            retryable=False,
        )

    if not pref.per_tool_channel_overrides_json:
        return None
    try:
        overrides = json.loads(pref.per_tool_channel_overrides_json)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    tool_overrides = overrides.get(tool_name) or {}
    explicit = tool_overrides.get(channel)
    if explicit is False:
        return ConnectorToolError(
            message=(
                f"User has disabled {tool_name!r} on channel {channel!r}."
            ),
            retryable=False,
        )
    if explicit is True:
        return _PREF_EXPLICIT_ALLOW
    # explicit is None: no opinion, defer to manifest.
    return None


def _resolve_channel_policy(
    manifest_tool: Any,
    channel: str,
) -> Optional[ConnectorToolError]:
    """Returns None when the tool is allowed on the channel, or a
    ConnectorToolError when it's denied. Per-user overrides
    (`ConnectorUserPreference`) land in T2c — until then, resolution
    is purely manifest-based.

    Order (matches architecture §4.4):
      3. Manifest channel_policy.deny → reject
      4. Manifest mutates: true AND channel in {voice, telegram} → reject
      5. Default

    A channel that STAGES rather than denies (`_MUTATES_CONFIRM_CHANNELS`,
    which R38 added `automation_thread` to) keeps its mutating tools on
    both passes — the first, where the elevation gate turns the call into
    a card, and the re-entry, where the user has tapped approve. Denying
    at this layer would make the staging unreachable on the first pass and
    the tap unhonourable on the second: a button that reports a failure
    for every write it offered, which is worse than the deny it replaced.

    That is decided by the CHANNEL alone, above. This function took an
    `approved_action_id` for a while and never read it, while its
    docstring said the parameter was what made re-entry safe and that the
    agent path could not lift its own gate through it — both false, and
    exactly the shape that gets "simplified" wrongly later. The real gate
    is `_needs_card`, which forces a card for anything that mutates here.
    """
    cp = manifest_tool.channel_policy
    if channel in cp.deny:
        return ConnectorToolError(
            message=f"tool {manifest_tool.name!r} is not available on channel {channel!r}",
            retryable=False,
        )
    _deny_channels = _MUTATES_DEFAULT_DENY_CHANNELS
    from app.config import settings as _settings
    if getattr(_settings, "injection_fencing_v2", False):
        _deny_channels = _MUTATES_DEFAULT_DENY_CHANNELS | _MUTATES_UNATTENDED_DENY_CHANNELS
    if channel in _MUTATES_CONFIRM_CHANNELS:
        # This channel does not deny a mutating tool — it STAGES one
        # (the elevation gate below forces a card for anything that
        # mutates here) and executes it on the user's tap. Denying at
        # this layer would make the staging unreachable, and denying
        # the re-entry would make the tap unhonourable.
        _deny_channels = _deny_channels - _MUTATES_CONFIRM_CHANNELS
    if manifest_tool.mutates and channel in _deny_channels:
        return ConnectorToolError(
            message=(
                f"mutating tool {manifest_tool.name!r} is denied by default "
                f"on {channel!r} (no inline confirmation surface). "
                f"User must invoke from web."
            ),
            retryable=False,
        )
    if cp.default == "deny":
        return ConnectorToolError(
            message=f"tool {manifest_tool.name!r} default-denied on {channel!r}",
            retryable=False,
        )
    return None


# ─── Refresh coordination ────────────────────────────────────────────


def _needs_refresh(identity) -> bool:
    """True when the access token expires within the skew window. False
    when there's no expiry (some providers don't set one; defer to
    provider's tool-call-time 401 to detect death)."""
    if identity.access_expires_at is None:
        return False
    return identity.access_expires_at < datetime.utcnow() + timedelta(
        seconds=_REFRESH_SKEW_SECONDS,
    )


async def _refresh_with_coalescing(
    db: AsyncSession,
    entry,
    identity,
) -> tuple[Any, Optional[ConnectorReauthRequired]]:
    """Refresh the identity's access token under a per-identity lock so
    50 concurrent callers coalesce to ONE provider.refresh().

    Returns `(refreshed_identity, None)` on success or
    `(stale_identity, ConnectorReauthRequired)` on failure.
    """
    if not identity.refresh_token:
        # No refresh_token → can't refresh. Mark for re-auth.
        await vault.mark_reauth_required(
            db, identity.id,
            reason="access token expired and no refresh_token available",
        )
        return identity, ConnectorReauthRequired(
            reauth_url=f"/agent/integrations/{identity.connector_id}",
        )

    async with _lock_for(identity.id):
        # Re-fetch and re-check inside the lock — another caller may
        # have already refreshed while we were waiting. This is the
        # coalescing point.
        latest = await vault.get(db, identity.user_id, identity.connector_id)
        if latest is None:
            # Disconnected mid-call. Treat as reauth.
            return identity, ConnectorReauthRequired(
                reauth_url=f"/agent/integrations/{identity.connector_id}",
            )
        if not _needs_refresh(latest):
            # Someone else refreshed first. Use their result.
            return latest, None

        try:
            # `latest.scopes` is what the user actually consented to, and
            # it is the identity's rather than the manifest's on purpose
            # — Entra reissues only already-consented scopes, so feeding
            # a grown manifest back would break every existing identity
            # at once. See `_microsoft_base._refresh_scope_param`.
            new_tokens = await entry.provider.refresh(
                latest.refresh_token, scopes=latest.scopes,
            )
        except RefreshFailed as e:
            await vault.mark_reauth_required(
                db, latest.id, reason=f"provider.refresh raised RefreshFailed: {e}",
                event_type=EVENT_REFRESH_FAILED,
            )
            return latest, ConnectorReauthRequired(
                reauth_url=f"/agent/integrations/{latest.connector_id}",
            )
        except Exception as e:
            # Unexpected — treat as reauth-required so the user can
            # recover, but log loudly.
            logger.exception(
                "[dispatcher] provider.refresh raised unexpectedly: %s", e,
            )
            await vault.mark_reauth_required(
                db, latest.id,
                reason=f"unexpected refresh exception: {type(e).__name__}: {e}",
                event_type=EVENT_REFRESH_FAILED,
            )
            return latest, ConnectorReauthRequired(
                reauth_url=f"/agent/integrations/{latest.connector_id}",
            )

        # Persist via vault.update — handles its own audit-then-act.
        await vault.update(
            db,
            latest.id,
            access_token=new_tokens.access_token,
            refresh_token=new_tokens.refresh_token,
            access_expires_at=new_tokens.expires_at,
            event_type=EVENT_REFRESH_SUCCEEDED,
        )
        # Re-fetch the now-fresh struct.
        refreshed = await vault.get(db, latest.user_id, latest.connector_id)
        if refreshed is None:
            # Pathological — the vault row vanished between update and
            # re-read. Treat as reauth.
            return identity, ConnectorReauthRequired(
                reauth_url=f"/agent/integrations/{latest.connector_id}",
            )
        return refreshed, None


# ─── Elevation staging ───────────────────────────────────────────────


def summarize_pending_action(tool_name: str, payload: dict) -> str:
    """One line describing what the user is being asked to approve.

    Rendered as the card's title, and it is also the ONLY description a
    text-only surface (a notification, an audit export, a screen reader
    reaching the card's aria-label) ever gets. So it names the outcome
    and the recipient — "Send email to sam@x.com" — never the tool.

    Falls back to the tool name for connectors that have not earned a
    bespoke line yet; a generic card is still a real gate.
    """
    def _s(key: str) -> str:
        v = payload.get(key)
        return v.strip() if isinstance(v, str) else ""

    # Shape first, name second. Anything carrying a recipient AND a
    # subject is mail, whoever delivers it — so a connector added later
    # gets a real card line instead of "Run fastmail__send_message"
    # without anyone remembering to extend this list.
    if _s("to") or "subject" in payload:
        to = _s("to") or "(no recipient)"
        subject = _s("subject") or "(no subject)"
        return f"Send email to {to} — “{subject}”"
    if tool_name == "linkedin__share_post":
        return "Post publicly to your LinkedIn feed"
    if tool_name == "calendar__create_event":
        title = _s("summary") or _s("title") or "(untitled)"
        when = _s("start") or _s("start_time")
        return f"Create calendar event “{title}”" + (f" at {when}" if when else "")
    if tool_name == "github__create_comment":
        owner, repo = _s("owner"), _s("repo")
        number = payload.get("number")
        where = f" on {owner}/{repo}#{number}" if owner and repo and number else ""
        return f"Post a GitHub comment{where}"
    if tool_name.startswith("sheets__"):
        return f"Write to your spreadsheet ({tool_name.split('__', 1)[1]})"
    if tool_name.startswith("drive__"):
        return f"Write to your Drive ({tool_name.split('__', 1)[1]})"
    return f"Run {tool_name}"


async def _stage_pending_action(
    db: AsyncSession,
    *,
    user_id: str,
    connector_id: str,
    tool_name: str,
    tool_input: dict,
    channel: str,
    agent_request_id: Optional[str],
    manifest_tool,
) -> ConnectorResult:
    """Persist the draft and return the confirmation result.

    Dedupes against an identical live draft: a model that re-issues the
    same send after reading "awaiting confirmation" — which they do —
    must not stack a second identical card on the thread. Same user,
    same tool, same arguments, still pending and unexpired → hand back
    the EXISTING action_id.

    A staging failure is fail-CLOSED. If we cannot record the draft we
    return an error rather than falling through to the provider: the
    entire point of the gate is that this call does not happen without
    a recorded, user-visible approval.
    """
    from app.db.models import ConnectorPendingAction

    now = datetime.utcnow()
    expires_at = now + timedelta(hours=_PENDING_ACTION_TTL_HOURS)
    payload_json = json.dumps(tool_input, sort_keys=True, default=str)
    summary = summarize_pending_action(tool_name, tool_input)

    try:
        from sqlalchemy import select as _select

        existing = (await db.execute(
            _select(ConnectorPendingAction)
            .where(ConnectorPendingAction.user_id == user_id)
            .where(ConnectorPendingAction.tool_name == tool_name)
            .where(ConnectorPendingAction.status == "pending")
            .where(ConnectorPendingAction.payload_json == payload_json)
            .where(ConnectorPendingAction.expires_at > now)
            .order_by(ConnectorPendingAction.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if existing is not None:
            return ConnectorConfirmationRequired(
                action_id=existing.id,
                summary=summary,
                payload=dict(tool_input),
                expires_at=existing.expires_at,
            )

        row = ConnectorPendingAction(
            user_id=user_id,
            connector_id=connector_id,
            tool_name=tool_name,
            payload_json=payload_json,
            status="pending",
            channel=channel,
            agent_request_id=agent_request_id,
            created_at=now,
            expires_at=expires_at,
        )
        db.add(row)
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.exception(
            "[dispatcher] could not stage pending action for tool=%s: %s",
            tool_name, e,
        )
        return ConnectorToolError(
            message=(
                "Could not stage this action for your confirmation, so it "
                "was not run. Try again in a moment."
            ),
            retryable=True,
        )

    # Audit AFTER the row exists so the event can name the action id.
    # Best-effort: the draft is already durable and the user can act on
    # it: losing the audit line must not cost them the card.
    try:
        await _audit_then_commit(
            db,
            user_id=user_id,
            connector_id=connector_id,
            event_type=EVENT_TOOL_ELEVATION_REQUIRED,
            channel=channel,
            tool_name=tool_name,
            agent_request_id=agent_request_id,
            metadata={
                "action_id": row.id,
                "summary": summary,
                "input": _redact(tool_input, manifest_tool.output_redaction),
            },
        )
    except Exception:
        logger.exception(
            "[dispatcher] elevation audit failed for action=%s (card still live)",
            row.id,
        )

    return ConnectorConfirmationRequired(
        action_id=row.id,
        summary=summary,
        payload=dict(tool_input),
        expires_at=expires_at,
    )


# ─── Outcome audit ───────────────────────────────────────────────────


async def _audit_outcome(
    db: AsyncSession,
    *,
    user_id: str,
    connector_id: str,
    tool_name: str,
    channel: str,
    agent_request_id: Optional[str],
    success: bool,
    metadata: dict,
) -> None:
    """Audit the post-execute outcome. NOT fail-closed — if this audit
    fails, we still return the result to the caller (the tool call did
    succeed; only the audit row is missing). The AuditError is logged
    but swallowed."""
    try:
        await _audit_then_commit(
            db,
            user_id=user_id,
            connector_id=connector_id,
            event_type=EVENT_TOOL_SUCCEEDED if success else EVENT_TOOL_FAILED,
            channel=channel,
            tool_name=tool_name,
            agent_request_id=agent_request_id,
            metadata=metadata,
        )
    except VaultAuditError as e:
        logger.error(
            "[dispatcher] post-execute audit failed for user=%s tool=%s — "
            "result was already returned to caller: %s",
            _hash_user(user_id), tool_name, e,
        )


def _outcome_metadata(
    result: ConnectorResult,
    output_redaction: list[str],
) -> dict:
    """Build the audit metadata for a tool outcome. Different shape per
    result variant; sensitive fields stripped per output_redaction."""
    if isinstance(result, ConnectorOk):
        return {"output": _redact_string_blob(result.content, output_redaction)}
    if isinstance(result, ConnectorRateLimited):
        return {"variant": "rate_limited", "retry_after_s": result.retry_after_s}
    if isinstance(result, ConnectorReauthRequired):
        return {"variant": "reauth_required", "reauth_url": result.reauth_url}
    if isinstance(result, ConnectorProviderDown):
        return {
            "variant": "provider_down",
            "provider_status_url": result.provider_status_url,
        }
    if isinstance(result, ConnectorScopeMissing):
        return {"variant": "scope_missing", "required_scope": result.required_scope}
    if isinstance(result, ConnectorToolError):
        return {
            "variant": "tool_error",
            "message": result.message[:300],
            "retryable": result.retryable,
        }
    return {"variant": "unknown", "type": type(result).__name__}


# ─── Output redaction ────────────────────────────────────────────────


def _redact(payload: dict, fields: list[str]) -> dict:
    """Strip top-level keys named in `fields` from a dict. Used for both
    input and output metadata in audit rows.

    Why one list for both: most sensitive field names (`body`, `snippet`,
    `attachments`) are sensitive in BOTH directions. Sending an email and
    reading one both expose `body`. A single `output_redaction` list per
    tool covers both with no duplication."""
    if not fields or not isinstance(payload, dict):
        return payload
    return {k: v for k, v in payload.items() if k not in fields}


def _redact_string_blob(blob: str, fields: list[str]) -> Any:
    """The provider's ConnectorOk content is a string — typically
    JSON-encoded. Try to parse + redact + re-emit; on parse failure
    record a length-bounded repr (NEVER the raw blob, since it might
    contain sensitive data we don't have field names for)."""
    if not blob:
        return None
    if not fields:
        # No redaction declared — record a short repr only.
        return {"output_repr": blob[:200] + ("..." if len(blob) > 200 else "")}
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {
            "output_repr": blob[:200] + ("..." if len(blob) > 200 else ""),
            "redaction_skipped_reason": "output_not_json",
        }
    if isinstance(parsed, dict):
        return _redact(parsed, fields)
    return {"output_repr": str(parsed)[:200]}


# ─── Logging helpers ─────────────────────────────────────────────────


def _hash_user(user_id: str) -> str:
    """First 8 hex chars of sha256(user_id). NEVER log the raw user_id —
    treats it as PII for our log streams."""
    return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:8]


def _log(
    user_hash: str,
    connector_id: str,
    tool_name: str,
    channel: str,
    outcome: str,
    started_at: float,
    *,
    level: str = "INFO",
    phases: str = "",
) -> None:
    duration_ms = int((time.monotonic() - started_at) * 1000)
    msg = (
        f"[dispatcher] user={user_hash} connector={connector_id} "
        f"tool={tool_name} channel={channel} outcome={outcome} "
        f"duration_ms={duration_ms}"
    )
    if phases:
        # Phase breakdown: `preflight=XXms audit_in=YYms provider=ZZms
        # audit_out=AAms` — lets ops attribute tail latency to a phase
        # without rerunning with extra instrumentation.
        msg += f" phases=[{phases}]"
    if level == "WARNING":
        logger.warning(msg)
    elif level == "ERROR":
        logger.error(msg)
    else:
        logger.info(msg)


# TODO(future): per-tool rate-limit ENFORCEMENT. `manifest_tool.rate_limit`
# is informational in v1; T5a (observability) or a dedicated follow-up
# will own the actual token-bucket / sliding-window logic. The natural
# integration point is just before the audit-then-act gate in step 5
# above — return ConnectorRateLimited(retry_after_s=...) and skip the
# rest of the flow.


def reset_locks_for_tests() -> None:
    """Drop the per-identity lock dict. Tests only — production never
    needs to reset (locks are bounded by active identities)."""
    _refresh_locks.clear()


# Public aliases for the refresh helpers — see the rationale block at
# the top of this module. Bound here (not at definition site) so the
# `_needs_refresh` / `_refresh_with_coalescing` symbols exist first.
needs_refresh = _needs_refresh
refresh_with_coalescing = _refresh_with_coalescing
