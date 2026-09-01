"""Delivery — the brief reaching the user where they asked for it.

CONTRACT-R43 §1.2/§8. An automation's `delivery` block names channels,
a format and a cadence; this module is the only thing that acts on it.

Three rules hold the whole file up, and every refusal below is one of
them saying no:

  1. **Every channel is the USER.** A DM to themselves, a draft in their
     own mailbox, a hold with no attendees. Nothing here can address
     anyone else — not because the copy says so, but because the target
     of every write is checked against the connected account's own
     identity before the row is staged. §1.3.

     That check is a POSITIVE proof and it is total:
     `_check_addressed_to_the_user` has a branch for every id in
     `_CONNECTOR_CHANNELS` and ends in a raise, so a tenth channel
     refuses until somebody writes its proof. It used to fall off the
     end with a bare `return`, which meant `teams_chat`, `notion_page`
     and `calendar_hold` were checked for nothing but a non-empty
     target — a grant pinned to a group chat or a team page is a
     perfectly legal grant for a write STEP, and those three would have
     delivered the user's whole ranked brief into it. §6.4's
     reassurance line was true for two connectors out of five.

     Where a provider exposes no way to prove it (`notion_page`), the
     channel is named in `UNVERIFIABLE_CHANNELS` and REFUSED rather
     than passed. §0.2: an option that cannot be honoured is not
     offered, and the words that say why live beside the id so the
     delivery picker can print them.

  2. **Every write goes through the outbox.** Same staging, same undo
     window, same grant gate, same `automation_writes` row and the same
     failed-write turn as a write STEP. A delivery is not a special kind
     of write; it is a write the user configured somewhere else.

  3. **A channel that cannot be written to fails VISIBLY.** The run's
     thread gets the same `ok:false` tool turn a failed step gets, with
     the reason in the string table's words, and the job config carries
     a per-channel record. A brief that quietly did not arrive is worse
     than one that arrived late: the user has no way to find out.

`app` is deliberately a no-op here — the thread turn and its push
already happened, and posting again would double the one channel every
user has on by default (§1.5).

WhatsApp and Telegram are the agent's OWN channels rather than a
connected third-party account, so they go through the existing
`routines.channel_dispatcher` (which owns the recipient resolution and
the per-adapter formatting) instead of the outbox. There is no grant to
check and nothing to undo: the agent is messaging its owner from its own
number.
"""

from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from app.db.models import AutomationOutbox

from . import catalog
from .brief_render import Brief, PdfUnavailable, brief_render

logger = logging.getLogger(__name__)

#: §1.4's third cadence, REMOVED from `catalog.CADENCES` in R43 wave
#: three. It promised a batched send and there was no batcher — the only
#: collector in the engine is the run itself — so picking it delivered
#: WITH the run and logged a warning: a picker that does not do what it
#: says, which §0.2 forbids as plainly as a chip that narrows nothing.
#:
#: Implementing it was considered and does not fit the existing routine
#: machinery: a batcher needs a store of undelivered briefs, a sweep
#: that owns the hour boundary, its own idempotency (the outbox key is
#: `run:<job_id>`, and an hourly send has no job), and an answer for
#: what a batch of one is. That is a feature, not an integration edit.
#:
#: The flag stays as the name of the gap. `deliver_brief` no longer
#: reads it, because `hourly` can no longer be stored: `catalog` is the
#: closed table `validate_delivery` and `PUT /delivery` both check
#: against, so an automation that asked for it now gets
#: `unknown_cadence` at the write instead of a promise at the read.
HOURLY_BATCHER = False

#: The channels this module writes through a connected account. `app`,
#: `whatsapp` and `telegram` are handled above/below it and are
#: deliberately absent — every id here needs a grant and a target.
#:
#: `target` is NOT chosen by this file. It is the pinned target of the
#: approved grant backing the write, which is what the dispatcher checks
#: the call against (`connector_dispatcher._resolve_automation_grant`);
#: choosing our own would produce a row the platform refuses at fire
#: time, after the run has told the user it delivered.
_CONNECTOR_CHANNELS: dict[str, dict] = {
    "slack_dm": {"connector_id": "slack", "tool": "slack__send_message",
                 "param": "channel", "body_param": "text"},
    "teams_chat": {"connector_id": "teams",
                   "tool": "teams__send_chat_message",
                   "param": "chat_id", "body_param": "message"},
    "gmail_draft": {"connector_id": "gmail", "tool": "gmail__create_draft",
                    "param": "to", "body_param": "body"},
    "outlook_mail": {"connector_id": "outlook",
                     "tool": "outlook__create_draft",
                     "param": "to", "body_param": "body"},
    # R43 — `notion__append_blocks`, not `notion__create_page`. §1.2
    # promises "appended under today's date" and a create makes a CHILD
    # PAGE: one page per day per automation, none of them the page the
    # user pinned. The append tool declares `page_id` as its pinned
    # target, so the grant covers exactly the page they picked.
    "notion_page": {"connector_id": "notion", "tool": "notion__append_blocks",
                    "param": "page_id", "body_param": "content"},
    # R43 — `param` was None while the calendar manifest declared no
    # `target_param_by_action` entry, so no grant could cover the call
    # and this channel refused 100% of the time. It declares
    # `calendar_id` now, and the id MUST be passed explicitly even
    # though the tool schema defaults it: `connector_dispatcher`
    # compares `tool_input[target_param]` against the pinned grant id,
    # and a schema default is not an argument.
    "calendar_hold": {"connector_id": "calendar",
                      "tool": "calendar__create_event",
                      "param": "calendar_id", "body_param": "description"},
}

#: Channels this module cannot PROVE reach the user and nobody else,
#: and the words that say why. Read by `workflow._channel_state`, which
#: serves them as `available: false` + `reason` so the delivery sheet
#: never lights a control that refuses at use time (§0.2).
#:
#: `notion_page` is the whole list. Notion's public API exposes no
#: sharing state at all: a page object carries `parent`, `created_by`
#: and `last_edited_by`, and nothing anywhere says who else can read
#: it. A page shared with a teamspace and a page nobody else has ever
#: seen are byte-identical from here, so "this brief reaches you alone"
#: is a claim this module would be guessing at — and the thing it would
#: leak is the user's mail subjects, who is waiting on them, and their
#: board. Guessing is not one of rule 1's options.
#:
#: The channel keeps its row in `_CONNECTOR_CHANNELS` on purpose: the
#: writer, the append tool and the dated heading are correct and stay
#: tested, so the day Notion ships a permissions read this is one
#: deletion away from working.
UNVERIFIABLE_CHANNELS: dict[str, str] = {
    "notion_page": (
        "Notion cannot say who else can read a page, so it will not put "
        "your brief on one"
    ),
}

#: A Slack conversation id that is a DM. `D…` is a direct message and
#: `U…`/`W…` is a person (posting to which OPENS their DM) — and for a
#: user token that person is the token's own owner, checked below.
#: `C…`/`G…` are a channel and a group: other people, and refused.
_SLACK_DM_PREFIXES = ("D", "U", "W")

#: A 15-minute block to read the brief in (§1.2) — a hold, never an
#: invite, which is why `attendees` is an empty list and stays one.
CALENDAR_HOLD_MINUTES = 15


#: A refusal this file raised → the verb-dictionary row that describes
#: it honestly. Anything not here is a write that did not happen.
_REFUSAL_VERBS = {"grant_missing": "scope_missing"}

#: WhatsApp / Telegram failure reasons → the sentence that reaches the
#: thread. `channel_dispatcher` answers in tokens (`no_recipient`,
#: `no_adapter`) or an exception class name; none of those is something
#: to put in front of a person, and the ledger's grammar would drop the
#: turn rather than serve one. Anything unlisted falls back to the
#: channel's own name, which is at least true.
_OWN_CHANNEL_SENTENCES = {
    "no_recipient": "It does not have a number to send yours to yet — "
                    "link the channel again",
    "no_adapter": "That channel is not connected on this agent",
}


class DeliveryRefused(Exception):
    """One channel cannot be written to, and why — in a reason code the
    string table already has words for.

    `detail` is for the LOG and may name a tool; `sentence` is the half
    that reaches the thread and may not. The ledger's grammar refuses a
    raw tool id in any served string (`_RAW_TOOL_RE`), and it caught
    exactly that here: the first version of this refusal put
    "no approved permission backs slack__send_message" in front of the
    user, and the turn was dropped — so a delivery that failed visibly
    by design failed silently in practice.
    """

    def __init__(self, reason_code: str, detail: str = "",
                 sentence: str = ""):
        super().__init__(detail or reason_code)
        self.reason_code = reason_code
        self.detail = detail
        self.sentence = sentence


# ── what the run is delivering, and where ────────────────────────────


def effective_delivery(delivery: Any, source: Any = None) -> dict:
    """`{channels, format, cadence}` for THIS run (§8).

    A per-connector ping overrides the automation's default, and only
    for a run the connector's own instant trigger started — that is the
    whole point of the override: "when JIRA wakes you, wake you HERE".
    A scheduled run has no connector behind it and keeps the default.

    An override names ONE channel, never a list: §5.4's picker is
    single-select on both halves, and a ping that fanned out to
    everything would be the default with extra steps.
    """
    d = delivery if isinstance(delivery, dict) else {}
    out = {
        "channels": catalog.order_channels(
            d.get("channels") or catalog.DEFAULT_DELIVERY["channels"]),
        "format": (d.get("format")
                   if catalog.is_format(d.get("format") or "")
                   else catalog.DEFAULT_DELIVERY["format"]),
        "cadence": (d.get("cadence")
                    if catalog.is_cadence(d.get("cadence") or "")
                    else catalog.DEFAULT_DELIVERY["cadence"]),
    }
    if source is None or getattr(source, "mode", "") == "schedule":
        return out
    ping_channel = getattr(source, "ping_channel", None)
    ping_format = getattr(source, "ping_format", None)
    if ping_channel and catalog.is_channel(ping_channel):
        out["channels"] = [ping_channel]
    if ping_format and catalog.is_format(ping_format):
        out["format"] = ping_format
    return out


async def _grant_for(
    user_id: str, automation, connector_id: str, tool: str,
) -> dict:
    """The APPROVED grant backing this delivery, with its pinned target.

    Fails closed and says which way it failed, because the two have
    different fixes: `grant_missing` means the user has not approved
    this channel yet (the delivery picker owes them the card), while an
    approved grant for a different tool means the automation is pointed
    at the wrong write.
    """
    from . import permissions, registry as _reg
    ids = permissions.write_grant_ids(automation, connector_id)
    for gid in ids:
        grant = await _reg.fetch_grant(user_id, gid) or {}
        if grant.get("status") != "approved":
            continue
        if grant.get("tool_name") != tool:
            continue
        target = grant.get("target") or {}
        if not isinstance(target, dict) or not target.get("id"):
            continue
        return {"id": gid, "target": target}
    raise DeliveryRefused(
        "grant_missing",
        f"no approved permission backs {tool} for this automation",
        sentence="You have not given it permission to write there yet",
    )


#: How a channel proves the target is the token's OWNER (§1.3).
#:
#: `oauth.py` writes `provider_account_id` from
#: `tokens.get("account") or tokens.get("login")` — Microsoft's field
#: and GitHub's. Slack's token payload puts the user's id under `id`
#: (its `token_lift_key: authed_user` overlays the USER block) and
#: Microsoft's token endpoint returns neither, so `account` is NULL for
#: every Slack and every Outlook identity ever connected. That is not
#: cosmetic here: `_check_addressed_to_the_user` refuses `outlook_mail`
#: with `unknown_account` whenever the address is empty, so the channel
#: was offered to every Outlook user and refused every one of them —
#: and `slack_dm` could not tell the owner's `U…` from a colleague's.
#:
#: `oauth.py` is not this round's file to change, and a backfill would
#: still leave existing connections NULL. So the address is resolved
#: from a LIVE source instead, which is the better answer anyway: it
#: reads the token that is about to be used rather than a column
#: written once at connect time.
async def _live_account(user_id: str, connector_id: str) -> str:
    """The connected account's own identity, live from the provider, or "".

    Empty is a real answer and every caller treats it as one: an
    unprovable owner is a REFUSAL, never a guess. Nothing here may
    raise — a delivery that cannot prove ownership declines; it does not
    fail the run.

    Slack answers `auth.test` through the connector's own public helper.
    Outlook has no public one, so this reaches for the two module-level
    functions the provider uses itself; both are wrapped, so the day
    either is renamed this degrades to "could not tell which mailbox is
    yours" — the same sentence the empty column produces today — rather
    than to an exception on the delivery path.

    Calendar answers with the id of the token owner's PRIMARY calendar,
    which is the calendar analogue of a mailbox address: Google names
    it after the account, and every other calendar the token can write
    to (`…@group.calendar.google.com`, a colleague who shared theirs) is
    somebody else's room. R43 — before this, `calendar_hold` had no
    identity to be measured against and no branch to measure it in.
    """
    try:
        if connector_id == "slack":
            from app.connectors.slack import provider as _slack
            ident = await _slack.self_identity_for_user(user_id) or {}
            return str(ident.get("user_id") or "")
        if connector_id == "outlook":
            from app.connectors.outlook import provider as _outlook
            token = await _outlook._resolve_token(user_id)
            return str(await _outlook._mailbox_address(user_id, token) or "")
        if connector_id == "calendar":
            from app.connectors._google_base import google_request
            from app.connectors.calendar import provider as _cal
            token = await _cal._resolve_token(user_id)
            entry = await google_request(
                "GET", f"{_cal.CAL_API_BASE}/users/me/calendarList/primary",
                access_token=token, scope_hint="calendar.readonly",
            )
            return str((entry or {}).get("id") or "")
    except Exception as e:  # noqa: BLE001 — an unproven owner declines
        logger.warning("[automations] %s identity unresolved: %s: %s",
                       connector_id, type(e).__name__, str(e)[:200])
    return ""


async def _account_for(user_id: str, connector_id: str) -> str:
    """The identity `_check_addressed_to_the_user` is measured against.

    The stored column FIRST — it is free, and for Gmail it is the right
    answer (`oauth._gmail_post_connect` backfills it) — then the live
    read for the two connectors whose column is structurally empty.
    """
    from . import registry as _reg
    state = await _reg.fetch_connection_state(user_id)
    account = str((state.get(connector_id) or {}).get("account") or "")
    if account:
        return account
    return await _live_account(user_id, connector_id)


async def _owned_chats(user_id: str) -> set[str]:
    """The Teams chats this connection can PROVE hold nobody but its
    owner — the user's chat with themselves, and only that.

    Teams has no `/me` for this connector (it holds `Chat.Read` and
    `ChatMessage.Send`; `User.Read` would need every existing
    connection re-consented), so "which member is me" is a DEDUCTION,
    and the teams provider already ships it as a closed one:
    `_identify_self` returns an id only when the member sets of the
    caller's chats intersect to exactly one person, and None with a
    reason otherwise. This reuses it rather than re-deriving it, for
    the same reason `_live_account` reaches into slack and outlook: the
    provider owns what its own ids mean.

    A chat is the user's own iff its member set is exactly `{self}`.
    A `oneOnOne` with a colleague is NOT — it reaches the user, and it
    reaches the colleague, which is the half rule 1 is about.

    Empty is a real answer and the caller treats it as a refusal, never
    as a pass. Nothing here may raise: an unprovable chat declines; it
    does not fail the run.
    """
    try:
        from app.connectors.teams import provider as _teams
        token = await _teams._resolve_token(user_id)
        page = await _teams._graph(
            "GET", f"{_teams.GRAPH_API}/me/chats",
            access_token=token, user_id=user_id, scope_hint="Chat.Read",
            params={"$top": 50, "$expand": "members"},
        )
        chats = [c for c in ((page or {}).get("value") or [])
                 if isinstance(c, dict)]
        self_id, why = _teams._identify_self(chats)
        if not self_id:
            logger.info("[automations] teams self unresolved: %s", why)
            return set()
        return {str(c.get("id") or "") for c in chats
                if _teams._member_ids(c) == {self_id} and c.get("id")}
    except Exception as e:  # noqa: BLE001 — an unproven owner declines
        logger.warning("[automations] teams chats unresolved: %s: %s",
                       type(e).__name__, str(e)[:200])
    return set()


async def _owned_targets(user_id: str, channel_id: str) -> set[str]:
    """Targets `_check_addressed_to_the_user` will accept for a channel
    whose proof is a LOOKUP rather than a comparison.

    Gmail, Outlook, Slack and Calendar each prove ownership by matching
    the target against one identity (`_account_for`). Teams cannot: a
    chat id is not an identity, and the only chat that reaches the user
    alone has to be found. Empty for every other channel, and empty is
    a refusal there too — the caller never treats "no set" as "any
    target".
    """
    if channel_id == "teams_chat":
        return await _owned_chats(user_id)
    return set()


def _check_addressed_to_the_user(
    channel_id: str, connector_id: str, target: str, account: str,
    owned: Optional[set[str]] = None,
) -> None:
    """§1.3, enforced rather than asserted.

    The one invariant that must hold whatever the picker, the grant flow
    or a future channel does: a DELIVERY reaches the user and nobody
    else. A grant is the user approving a target, but the user can
    approve a target that is somebody else — `#platform` is a perfectly
    legitimate grant for a write STEP — and the delivery node's whole
    promise (§6.4's reassurance line) is that this list is not that.

    `account` is what the caller could PROVE about the connection
    (`_account_for`), never what it assumed. `owned` is the same thing
    for a channel whose proof is a lookup (`_owned_targets`). An empty
    one of either is therefore a refusal and not a pass: this check
    fails closed in both directions.

    TOTAL, and that is the point. Every branch below either returns or
    raises, and the function ENDS in a raise — so an id with no branch
    refuses instead of falling through. The bare `return` that used to
    sit at the end is the whole of finding 16: it made `teams_chat`,
    `notion_page` and `calendar_hold` pass any grant target at all.
    """
    t = str(target or "").strip()
    if not t:
        raise DeliveryRefused(
            "no_target", "the permission pins no target",
            sentence="The permission it has pins no destination")
    if channel_id in UNVERIFIABLE_CHANNELS:
        raise DeliveryRefused(
            "unverifiable_channel",
            f"{connector_id} exposes no way to prove {t} is the user's alone",
            sentence=UNVERIFIABLE_CHANNELS[channel_id],
        )
    if channel_id in ("gmail_draft", "outlook_mail"):
        if not account:
            raise DeliveryRefused(
                "unknown_account",
                f"{connector_id} did not report which mailbox it is",
                sentence="It could not tell which mailbox is yours",
            )
        if t.lower() != str(account).lower():
            raise DeliveryRefused(
                "not_the_user",
                "a delivery draft is addressed to you and to nobody else",
                sentence="A delivery draft is addressed to you and to "
                         "nobody else",
            )
        return
    if channel_id == "slack_dm":
        if t[:1].upper() not in _SLACK_DM_PREFIXES:
            raise DeliveryRefused(
                "not_the_user",
                "a Slack DM goes to your own conversation, not a channel",
                sentence="A Slack DM goes to your own conversation, never "
                         "to a channel",
            )
        # A `U…`/`W…` id must be the token's OWN owner: posting to
        # anyone else's id opens a DM with THEM, which is exactly the
        # thing §1.3 forbids and looks identical from here.
        #
        # `and account` used to guard this, so an unresolved owner let
        # any `U…` through. That was not a choice — Slack's
        # `provider_account_id` is NULL for every identity ever
        # connected (`oauth.py` reads Microsoft's `account` and GitHub's
        # `login`, and Slack's token payload has neither), so requiring
        # it would have refused every Slack delivery on the platform.
        # `_account_for` resolves it live now, so the guard can be what
        # the invariant says: an owner nobody can prove is a REFUSAL.
        #
        # A `D…` target needs no such proof and keeps none: it is a
        # conversation out of this token's own DM list, so it is the
        # user's by construction.
        if t[:1].upper() in ("U", "W") and t != account:
            raise DeliveryRefused(
                "not_the_user",
                "that Slack id is not yours"
                if account else "this Slack account did not say who it is",
                sentence=("That Slack conversation is not yours" if account
                          else "It could not tell which Slack account is "
                               "yours"),
            )
        return
    if channel_id == "teams_chat":
        # The user's chat with THEMSELVES, proven by membership, or
        # nothing. A `oneOnOne` with a colleague reaches the user AND
        # the colleague; a group chat reaches everyone in it; and a
        # grant pinned to either is a legal grant for a write STEP, so
        # the grant cannot be the proof. `owned` is `_owned_chats`'
        # closed answer (see there) and an empty one is a refusal.
        if t in (owned or set()):
            return
        raise DeliveryRefused(
            "not_the_user",
            "that Teams chat is not the one you have with yourself"
            if owned else "this Teams account did not say which chat is "
                          "yours alone",
            sentence=("A Teams delivery goes to your own chat, never to one "
                      "with other people in it" if owned else
                      "It could not tell which Teams chat is yours alone"),
        )
    if channel_id == "calendar_hold":
        # A hold with no attendees still SITS somewhere, and everyone
        # who can read that calendar reads the brief in it. So the
        # calendar has to be the user's own, which is the same test
        # `gmail_draft` makes against a mailbox — Google names the
        # primary calendar after the account, and `primary` is the
        # literal alias for it.
        #
        # Everything else this token can write to is somebody else's
        # room: a `…@group.calendar.google.com` shared with the team, a
        # colleague who gave them write access.
        if t.lower() == "primary":
            return
        if not account:
            raise DeliveryRefused(
                "unknown_account",
                "calendar did not report which calendar is the user's own",
                sentence="It could not tell which calendar is yours",
            )
        if t.lower() != str(account).lower():
            raise DeliveryRefused(
                "not_the_user",
                "a delivery hold sits on your own calendar, not a shared one",
                sentence="A hold goes on your own calendar, never on a "
                         "shared one",
            )
        return
    # No branch means no proof. A tenth channel refuses here until
    # somebody writes one, rather than shipping unchecked.
    raise DeliveryRefused(
        "unverifiable_channel",
        f"no ownership check is written for {channel_id}",
        sentence="It cannot yet prove a brief sent there reaches only you",
    )


def _subject(automation, brief: Brief, now: datetime) -> str:
    name = str(getattr(automation, "name", "") or "Your brief")
    return f"{name} — {now.strftime('%-d %B')}"[:200]


def _payload_for(
    channel_id: str, *, target: str, brief: Brief, automation,
    now: datetime,
) -> dict:
    """The tool arguments for one channel's write."""
    spec = _CONNECTOR_CHANNELS[channel_id]
    if channel_id == "calendar_hold":
        end = now + timedelta(minutes=CALENDAR_HOLD_MINUTES)
        return {
            # EXPLICIT, though the tool schema defaults it to "primary":
            # `connector_dispatcher._resolve_automation_grant` compares
            # `tool_input[target_param]` against the grant's pinned id,
            # and a schema default is not an argument — it would refuse
            # with "the call targeted nothing instead".
            spec["param"]: target,
            "summary": _subject(automation, brief, now),
            "start": now.replace(microsecond=0).isoformat(),
            "end": end.replace(microsecond=0).isoformat(),
            "description": brief.text[:8000],
            # A HOLD, not an invite. Empty and it stays empty: the moment
            # this list has a name in it the block becomes a meeting
            # somebody else is now expected at.
            "attendees": [],
        }
    body = {spec["param"]: target, spec["body_param"]: brief.text[:8000]}
    if channel_id in ("gmail_draft", "outlook_mail"):
        body["subject"] = _subject(automation, brief, now)
        # "One-page PDF" naming a text blob is the same lie as a chip
        # that narrows nothing. Both draft tools take the SAME shape
        # (R43) — at most 3 files, 3 MB decoded in total, which is
        # Graph's inline-attachment ceiling and three orders of
        # magnitude above a rendered brief — so this file does not have
        # to know which mail channel it is writing to. A malformed
        # attachment makes the tool REFUSE, so a delivery either carries
        # the file or reports failed; it never silently posts a brief
        # without the document its format is named after.
        if brief.document:
            filename, mime, blob = brief.document
            body["attachments"] = [{
                "filename": filename,
                "content_type": mime,
                "content_base64": base64.b64encode(blob).decode("ascii"),
            }]
    if channel_id == "outlook_mail":
        # §1.2's own words: "mail to yourself, marked read".
        body["is_read"] = True
    if channel_id == "notion_page":
        # "Appended under today's date" (§1.2), and the heading is how
        # the date gets there: `notion__append_blocks` puts it above the
        # brief and turns each line of `content` into its own paragraph,
        # so the pinned page reads as one dated entry per run.
        body["heading"] = now.strftime("%Y-%m-%d")
    return body


# ── the fan-out ──────────────────────────────────────────────────────


async def deliver_brief(
    db, *, automation, job_id: str, thread, groups: Any, title: str = "",
    delivery: Any = None, source: Any = None, idem_prefix: str = "",
) -> dict:
    """Send the run's brief to every picked channel.

    Returns `{channel_id: {"status", "reason"}}` — `delivered`,
    `skipped` (the `app` channel, whose brief already landed) or
    `failed`. The same map is stamped on the job config, so a run's own
    record answers "did it reach me" without reading the thread.

    Best-effort as a whole and per channel: this runs AFTER the run's
    terminal, so nothing here may raise into the run, and one dead
    channel must never cost the others theirs.
    """
    plan = effective_delivery(delivery, source)
    out: dict[str, dict] = {}
    now = datetime.now(timezone.utc)

    try:
        brief = brief_render(
            groups, plan["format"], title=title,
            slug=str(getattr(automation, "name", "") or "brief"),
        )
    except PdfUnavailable as e:
        # Never markdown under the name PDF. Every channel fails with
        # one reason, and the user is told which format could not be
        # made rather than handed a different one.
        logger.warning("[automations] brief PDF unavailable: %s", e)
        for cid in plan["channels"]:
            out[cid] = {"status": "failed", "reason": "format_unavailable"}
            if cid == "app":
                # The thread already has the brief; only the FILE is
                # missing, and saying "your brief did not reach this
                # chat" under it would be the false half of rule 3.
                continue
            await _append_refused_turn(
                db, automation=automation, job_id=job_id, thread=thread,
                channel_id=cid,
                refusal=DeliveryRefused(
                    "format_unavailable", str(e)[:200],
                    sentence="The one-page PDF could not be made, and it "
                             "will not send a different file under that "
                             "name"),
            )
        await _record(db, automation=automation, job_id=job_id, thread=thread,
                      results=out, plan=plan)
        return out
    except Exception as e:  # noqa: BLE001 — a render fault is not a run fault
        logger.warning("[automations] brief render failed: %s", e)
        return out

    for cid in plan["channels"]:
        try:
            out[cid] = await _deliver_one(
                db, channel_id=cid, automation=automation, job_id=job_id,
                thread=thread, brief=brief, now=now,
                idem_prefix=idem_prefix or f"run:{job_id}",
            )
        except DeliveryRefused as e:
            out[cid] = {"status": "failed", "reason": e.reason_code}
            await _append_refused_turn(
                db, automation=automation, job_id=job_id, thread=thread,
                channel_id=cid, refusal=e,
            )
        except Exception as e:  # noqa: BLE001 — one channel, not all of them
            logger.warning("[automations] delivery to %s failed: %s: %s",
                           cid, type(e).__name__, str(e)[:200])
            out[cid] = {"status": "failed", "reason": "unknown_error"}
            # Rule 3 again. Nothing after `flush_row_when_due` can
            # raise, so an exception reaching here means the outbox
            # never staged this channel and never appended its own
            # turn — this is the only record the user can get.
            await _append_refused_turn(
                db, automation=automation, job_id=job_id, thread=thread,
                channel_id=cid,
                refusal=DeliveryRefused(
                    "unknown_error", f"{type(e).__name__}: {str(e)[:200]}",
                    sentence="Something went wrong sending it there"),
            )

    await _record(db, automation=automation, job_id=job_id, thread=thread,
                  results=out, plan=plan)
    return out


async def _deliver_one(
    db, *, channel_id: str, automation, job_id: str, thread, brief: Brief,
    now: datetime, idem_prefix: str,
) -> dict:
    if channel_id == "app":
        # Already delivered: the thread's result turn and its push. §1.2.
        return {"status": "skipped", "reason": "already_in_this_chat"}

    if channel_id in ("whatsapp", "telegram"):
        return await _deliver_own_channel(
            channel_id, automation=automation, brief=brief)

    spec = _CONNECTOR_CHANNELS.get(channel_id)
    if spec is None:
        raise DeliveryRefused(
            "unknown_channel", f"no writer for {channel_id}",
            sentence="It has no way to send a brief there")
    if spec["param"] is None:
        # A channel whose tool declares no pinned-target parameter: no
        # grant can cover it, so `connector_dispatcher.
        # _resolve_automation_grant` refuses the call at fire time —
        # after the run has told the user it delivered. Refusing HERE
        # says so in words instead. Every shipped channel names one as
        # of R43 (`calendar__create_event` gained `calendar_id`); this
        # stays as the gate a tenth channel has to pass.
        raise DeliveryRefused(
            "no_target_param",
            f"{spec['tool']} declares no pinned target, so no permission "
            "can cover it",
            sentence="No permission can cover that one yet",
        )

    grant = await _grant_for(
        automation.user_id, automation, spec["connector_id"], spec["tool"])
    account = await _account_for(automation.user_id, spec["connector_id"])
    owned = await _owned_targets(automation.user_id, channel_id)
    target = str(grant["target"].get("id") or "")
    _check_addressed_to_the_user(
        channel_id, spec["connector_id"], target, account, owned)

    payload = _payload_for(channel_id, target=target, brief=brief,
                           automation=automation, now=now)
    row = AutomationOutbox(
        user_id=automation.user_id,
        automation_id=automation.id,
        job_id=job_id,
        connector_id=spec["connector_id"],
        tool_name=spec["tool"],
        payload_json=json.dumps(payload, sort_keys=True, default=str),
        grant_id=grant["id"],
        # Keyed by CHANNEL, not by index: a resumed run re-delivers to
        # the same places, and the unique index makes the second attempt
        # a no-op rather than a second copy of the brief.
        idempotency_key=f"{idem_prefix}:d:{channel_id}"[:128],
        # No undo window. The run is already terminal by the time
        # delivery happens — there is no card left to press undo on, and
        # a staged row nobody flushes is a brief that never arrives.
        execute_after=datetime.utcnow(),
        display_json=json.dumps(
            {"what": _delivery_what(channel_id),
             "target": grant["target"].get("label") or target,
             "audience": "you", "reversible": True},
            sort_keys=True, default=str),
    )
    db.add(row)
    from sqlalchemy.exc import IntegrityError
    try:
        await db.flush()
        await db.commit()
    except IntegrityError:
        await db.rollback()
        # The unique (automation_id, idempotency_key) index did its job:
        # this channel already has this run's brief.
        return {"status": "skipped", "reason": "already_delivered"}

    from .outbox import flush_row_when_due
    status = await flush_row_when_due(db, row.id)
    if status == "executed":
        return {"status": "delivered", "reason": None}
    return {"status": "failed", "reason": status or "lost"}


def _delivery_what(channel_id: str) -> str:
    """The write ledger's phrase for this delivery.

    Drawn from the catalogue's own `land` clause ("as a Slack DM", "in
    your Teams chat"), so the ledger says what the delivery sheet's
    footer said it would — one table, both surfaces.
    """
    entry = catalog.channel(channel_id) or {}
    return f"Sent your brief {entry.get('land') or 'to you'}"[:200]


async def _deliver_own_channel(
    channel_id: str, *, automation, brief: Brief,
) -> dict:
    """WhatsApp / Telegram, through the existing routine dispatcher.

    Not the outbox: there is no connected third-party account, no grant
    to check and nothing to undo — the agent is messaging its owner from
    its own number, which is the same send `deliver_to_extra_channels`
    has always made for routines.
    """
    from app.db.database import async_session_maker
    from app.agent.routines.channel_dispatcher import (
        deliver_to_extra_channels_detailed,
    )
    detailed = await deliver_to_extra_channels_detailed(
        user_id=automation.user_id,
        delivery_channels=[channel_id],
        routine_name=str(getattr(automation, "name", "") or ""),
        content=brief.text,
        db_session_maker=async_session_maker,
    )
    entry = (detailed or {}).get(channel_id) or {}
    status = entry.get("status")
    if status == "delivered":
        return {"status": "delivered", "reason": None}
    # RAISES, and that is finding 17. These two channels used to
    # `return` a failed dict, and `deliver_brief`'s loop only appends
    # the thread turn on `DeliveryRefused` — so a dropped WhatsApp
    # session left the run's thread saying nothing at all and the brief
    # silently stopped arriving. Rule 3 does not have an exception for
    # the channels that skip the outbox: the outbox is what makes the
    # OTHER failures visible (`outbox._append_failed_write_turn`), and
    # a channel that has no outbox row needs this file to do it.
    reason = str(entry.get("reason") or entry.get("error_class")
                 or "no_adapter")
    name = (catalog.channel(channel_id) or {}).get("name") or channel_id
    raise DeliveryRefused(
        reason,
        f"{channel_id} delivery {status or 'failed'}: "
        f"{entry.get('error_detail') or reason}",
        sentence=_OWN_CHANNEL_SENTENCES.get(
            reason, f"{name} did not take it this time"),
    )


# ── the record ───────────────────────────────────────────────────────


async def _append_refused_turn(
    db, *, automation, job_id: str, thread, channel_id: str,
    refusal: DeliveryRefused,
) -> None:
    """A channel that could not be written to, in the thread (rule 3).

    The same `ok:false` write turn the outbox appends for a failed send
    (`outbox._append_failed_write_turn`), because it is the same event
    from the user's side: the brief did not reach a place they asked for
    it to reach. The account line carries the reason and, where the
    string table has one, the button that fixes it.
    """
    if thread is None:
        return
    try:
        from . import account_health as _ah, ledger as _ledger
        from app.services import automation_verbs as _verbs
        spec = _CONNECTOR_CHANNELS.get(channel_id) or {}
        cid = spec.get("connector_id") or ""
        name = catalog.channel(channel_id) or {}
        code = _ah.classify(refusal.reason_code, refusal.detail)
        state, fix = _ah.state_for_reason(code)
        # WhatsApp and Telegram are the agent's OWN channels: there is
        # no connected account behind them, so every string the health
        # table composes ("I could not read {Connector}") would be
        # about a connector that does not exist. They take the plain
        # fallback below, which names the channel and stops there.
        if not cid:
            state, fix = "connected", None
        # The dictionary is the only source of `action`/`detail`
        # (CONTRACTS-R30 §1), and its DEFAULT pair is "Could not
        # connect / it did not answer" — which is a claim about the
        # provider, and false for a refusal this file made without
        # calling one. `write_failed` says the true thing: nothing was
        # changed. A missing permission keeps its own row, because the
        # fix is the grant card rather than a retry.
        act = _verbs.failure_action(
            cid, _REFUSAL_VERBS.get(refusal.reason_code, "write_failed"))
        line = _ah.sentence_for(
            account_state=state, reason_code=code, connector_id=cid,
            name=_ah.display_of(cid),
        ) if cid else ""
        if not line:
            # A refusal this file raised (`not_the_user`, `no_target_param`)
            # is not an account-health state, so the table has no sentence
            # and `state_for_reason` answers its ("connected", "retry")
            # default. A Retry button under "that Slack id is not yours"
            # offers a fix that cannot work, so BOTH are dropped together.
            line = f"Your brief did not reach {name.get('name') or channel_id}."
            fix = None
        steps_land = name.get("land") or "to you"
        await _ledger.append_turn(
            db, user_id=automation.user_id, thread=thread, run_id=job_id,
            kind="tool",
            payload={
                "account_id": cid, "tool_kind": "write",
                "action": act["action"], "detail": act["detail"],
                "ok": False, "ms": 0,
                "steps": [
                    {"text": f"Asked to send your brief {steps_land}",
                     "ok": True},
                    # `sentence`, never `detail`: see DeliveryRefused.
                    {"text": (refusal.sentence or act["detail"]
                              or "It could not be sent"),
                     "ok": False},
                ],
                "actions": [], "items": [], "write_ids": [], "rest": "",
                "line": line, "tone": "warning",
                **({"fix": fix} if fix else {}),
                "reason_code": code,
            },
        )
    except Exception as e:  # noqa: BLE001 — a record never fails a run
        logger.warning("[automations] delivery refusal turn skipped: %s: %s",
                       type(e).__name__, e)


async def _record(
    db, *, automation, job_id: str, thread, results: dict, plan: dict,
) -> None:
    """Stamp the per-channel outcome on the run.

    On the JOB, beside `accounts_failed` and `failed_sources`, because
    every other "did this run do what it said" question is answered from
    there — the home card's meta, the notification flip and the per-
    source resume all read that config blob.
    """
    try:
        from .executor_v2 import merge_job_config
        await merge_job_config(
            db, job_id,
            delivery={"channels": plan["channels"],
                      "format": plan["format"],
                      "cadence": plan["cadence"],
                      "results": results},
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("[automations] delivery stamp skipped: %s", e)
