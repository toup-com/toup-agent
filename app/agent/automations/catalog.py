"""The delivery catalogue: channels, formats, cadence (CONTRACT-R43 §1).

One table, three readers, and that is the whole reason this module
exists as its own file. The workflow payload builder composes the
canvas labels from it, the `PUT /delivery` writer validates against it,
and the spec validator checks a per-connector ping override against it
— three places that must agree on nine strings each, in one fixed
order, or the canvas node says one thing and the sheet says another.

Everything here is PURE: no session, no request, no clock. Availability
is deliberately NOT modelled — whether Slack is connected is a fact
about the ACCOUNT, and it belongs to the payload builder that already
holds the account list. This file answers only "what is a Slack DM
called, what does it say under the name, and which connector does it
need".

Ids are frozen (CONTRACT-R43 §1): both repos hard-code these strings.
"""

from __future__ import annotations

from typing import Optional

#: Fixed order (§1.2). The app renders the delivery sheet in exactly
#: this sequence, so a reorder here reorders the sheet.
#:
#: `connector_id` is the account that must be connected before the
#: channel can be picked; None means it needs no account. `needs_link`
#: is the OTHER kind of unavailability — the account exists but the
#: user has not yet told the channel who they are — and its value is
#: the words the Link pill asks for.
CHANNELS: tuple[dict, ...] = (
    {"id": "app", "name": "This app",
     "meta": "a push, then the brief in this chat",
     "land": "in this chat",
     "connector_id": None, "needs_link": None},
    {"id": "whatsapp", "name": "WhatsApp",
     "meta": "a message from your agent's number",
     "land": "as a WhatsApp message",
     "connector_id": None, "needs_link": "your number"},
    {"id": "telegram", "name": "Telegram",
     "meta": "a message from your agent bot",
     "land": "as a Telegram message",
     "connector_id": None, "needs_link": "your Telegram"},
    {"id": "slack_dm", "name": "Slack DM",
     "meta": "a direct message from your agent",
     "land": "as a Slack DM",
     "connector_id": "slack", "needs_link": None},
    {"id": "teams_chat", "name": "Teams chat",
     "meta": "a message in your chat with it",
     "land": "in your Teams chat",
     "connector_id": "teams", "needs_link": None},
    {"id": "gmail_draft", "name": "Gmail draft",
     "meta": "written to you, never sent out",
     "land": "as a Gmail draft",
     "connector_id": "gmail", "needs_link": None},
    {"id": "outlook_mail", "name": "Outlook mail",
     "meta": "mail to yourself, marked read",
     "land": "as Outlook mail",
     "connector_id": "outlook", "needs_link": None},
    {"id": "notion_page", "name": "Notion page",
     "meta": "appended under today's date",
     "land": "on a Notion page",
     "connector_id": "notion", "needs_link": None},
    {"id": "calendar_hold", "name": "Calendar hold",
     "meta": "a 15-minute block to read it",
     "land": "as a calendar hold",
     "connector_id": "calendar", "needs_link": None},
)

#: Single-select (§1.3). `rail` is the canvas link label between the
#: agent and the delivery node; `noun` is the prose form, and the two
#: are separate strings because "1 list" cannot be read aloud in a
#: sentence and "a ranked list" cannot fit on a 94pt rail.
FORMATS: tuple[dict, ...] = (
    {"id": "lines", "name": "Five short lines",
     "meta": "read it without opening anything",
     "rail": "5 lines", "noun": "five short lines"},
    {"id": "ranked", "name": "Ranked list",
     "meta": "grouped by what breaks first",
     "rail": "1 list", "noun": "a ranked list"},
    {"id": "pdf", "name": "One-page PDF",
     "meta": "printable, keeps every link",
     "rail": "1 PDF", "noun": "a one-page PDF"},
    {"id": "markdown", "name": "Markdown file",
     "meta": "for your notes app or your repo",
     "rail": "1 file", "noun": "a markdown file"},
    {"id": "csv", "name": "CSV of items",
     "meta": "one row per thing it found",
     "rail": "1 sheet", "noun": "a CSV of items"},
)

#: §1.4 — when the brief itself is sent, as opposed to where.
#:
#: R43 wave three, RECORDED REMOVAL: the spec's third cadence,
#: `hourly` "Batched hourly", is not offered. There is no batcher — the
#: only collector in the engine is the run itself — so the picker
#: delivered WITH the run and logged a warning, which is a control that
#: does not do what its label says. §0.2 forbids exactly that: "if the
#: platform cannot honour a choice, the option is not offered".
#:
#: Not implemented rather than deferred, because a batcher is a feature
#: and not an integration edit: it needs somewhere to hold undelivered
#: briefs, a sweep that owns the hour boundary, an idempotency key that
#: is not `run:<job_id>` (an hourly send has no job), and a decision
#: about what a batch of one is. `deliver.HOURLY_BATCHER` stays as the
#: name of that gap.
#:
#: Removing it here is what makes it unreachable everywhere: this tuple
#: is the closed table `spec_v2.validate_delivery` and
#: `PUT /delivery` both check against, so `hourly` is now
#: `unknown_cadence` at the write rather than a promise at the read.
CADENCES: tuple[dict, ...] = (
    {"id": "run", "label": "With the run"},
    {"id": "instant", "label": "The moment it matches"},
)

#: §1.5. The spec's demo default is `['This app', 'Slack DM']`; seeding
#: `slack_dm` on a REAL automation would start DMing a user who never
#: asked for it, so `app` alone is the default and the popup shows the
#: rest.
DEFAULT_DELIVERY: dict = {
    "channels": ["app"],
    "format": "ranked",
    "cadence": "run",
}

#: §6.4, verbatim. True only because `delivery` is always the user
#: themselves — an automation that posts to a team channel does that as
#: a STEP, which is a different thing in a different part of the sheet
#: (§1.6).
REASSURANCE = (
    "Every channel here is you — nothing goes to anyone else, and mail "
    "is written as a draft, never sent."
)

#: The canvas chip row is 94pt wide at 18pt a chip (spec §4).
MAX_NODE_CHIPS = 4

_CHANNELS_BY_ID = {c["id"]: c for c in CHANNELS}
_FORMATS_BY_ID = {f["id"]: f for f in FORMATS}
_CADENCES_BY_ID = {c["id"]: c for c in CADENCES}

CHANNEL_IDS: tuple[str, ...] = tuple(c["id"] for c in CHANNELS)
FORMAT_IDS: tuple[str, ...] = tuple(f["id"] for f in FORMATS)
CADENCE_IDS: tuple[str, ...] = tuple(c["id"] for c in CADENCES)


def channel(channel_id: str) -> Optional[dict]:
    """The channel entry, or None. Never a KeyError: every caller here
    is on a path where an unknown id is a 400 with words, not a 500."""
    return _CHANNELS_BY_ID.get(str(channel_id or ""))


def format_(format_id: str) -> Optional[dict]:
    """The format entry, or None. Trailing underscore because `format`
    is a builtin and this module is imported bare into label code."""
    return _FORMATS_BY_ID.get(str(format_id or ""))


def cadence(cadence_id: str) -> Optional[dict]:
    return _CADENCES_BY_ID.get(str(cadence_id or ""))


def is_channel(channel_id: str) -> bool:
    return str(channel_id or "") in _CHANNELS_BY_ID


def is_format(format_id: str) -> bool:
    return str(format_id or "") in _FORMATS_BY_ID


def is_cadence(cadence_id: str) -> bool:
    return str(cadence_id or "") in _CADENCES_BY_ID


def order_channels(channel_ids) -> list[str]:
    """The picked ids in CATALOGUE order, deduped, unknowns dropped.

    Order is the table's and never the caller's, for the same reason
    `validate_filters` canonicalises its list: two deliveries that reach
    the same person the same way must serialize identically, or the
    canvas chip row reshuffles itself after an unrelated edit.
    """
    wanted = {str(c) for c in (channel_ids or []) if isinstance(c, str)}
    return [cid for cid in CHANNEL_IDS if cid in wanted]


def _names(channel_ids) -> list[str]:
    return [_CHANNELS_BY_ID[cid]["name"] for cid in order_channels(channel_ids)]


def node_label(channel_ids) -> str:
    """The delivery node's title: the first channel, then "+N" (§2.1).

    "Nowhere yet" is the zero case rather than an empty node — a node
    with no title reads as a rendering fault, and the user CAN reach
    this state by unticking the last channel.
    """
    names = _names(channel_ids)
    if not names:
        return "Nowhere yet"
    extra = len(names) - 1
    return f"{names[0]} +{extra}" if extra else names[0]


def node_sub(format_id: str) -> str:
    """The line under the node: "as a ranked list", "as five short
    lines". The article lives in the noun, so this is "as " + noun —
    §2.1's inline comment says "as a " + noun, which would double the
    article for every format; the VALUE in that same block is what this
    matches."""
    fmt = format_(format_id) or format_(DEFAULT_DELIVERY["format"])
    return f"as {fmt['noun']}"


def node_chips(channel_ids) -> list[str]:
    """The canvas chip row: picked ids in catalogue order, at most four."""
    return order_channels(channel_ids)[:MAX_NODE_CHIPS]


def rail(format_id: str) -> str:
    """The right-hand link label between the agent and delivery."""
    fmt = format_(format_id) or format_(DEFAULT_DELIVERY["format"])
    return fmt["rail"]


def done_label(channel_ids, format_id: str) -> str:
    """The delivery sheet's sticky footer button (§6.5).

    Every channel is named, not just the first: this is the last thing
    the user reads before dismissing the sheet, and "This app +2" there
    would hide exactly what they came to check.
    """
    names = _names(channel_ids)
    fmt = format_(format_id) or format_(DEFAULT_DELIVERY["format"])
    where = " + ".join(names) if names else "nowhere yet"
    return f"Done — {where} · {fmt['noun']}"
