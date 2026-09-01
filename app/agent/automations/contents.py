"""What is actually INSIDE one account (R38).

The canvas draws a node per account and, until now, could say only what
the automation is allowed to do there. Tapping the node told you about
the permission, never about the material — so "start it at the right
place" was a decision made blind, and the pin (`focus`) it leads to had
nothing to pick from.

`account_contents` answers the other half: a uniform envelope of what
that account holds right now, dispatched per connector through the same
grant-gated platform RPC every automation read uses
(`registry.dispatch_via_platform`) — never a second HTTP client and
never a provider import.

    {account_id, connector_id, name, ok, reason,
     groups: [{key, label, kind, reason, items: [{id,kind,title,sub,at}]}],
     count, truncated, focus}

The one rule this module exists to hold: **absent is not empty.** An
unreachable agent, a dead credential, a provider having a bad minute
and an account that genuinely holds nothing are four different facts,
and the previous generation of this repo shipped all four as `[]` —
"No playlists yet" for a full library, read by the founder as data
loss. So `ok` is a boolean the caller must branch on, `reason` carries
the sentence, and a group that failed says so IN the group rather than
vanishing from the list. A group with `items: []` and `reason: null` is
the only shape that means "there is nothing here".

R43 adds two things on top of that, both in CONTRACT-R43.

`account_sources` (§5) enumerates the account's REAL objects — the
channels you are in, the projects you have tickets on, the repositories
you push to — so the popup's "what it may open here" list is the
account rather than a seed table. Every source carries a `kind` from
`spec.FOCUS_KINDS`, because picking one becomes a pin descriptor and
R39 proved an invented kind 409s every write.

And the envelope gains the fields the design's row actually needs (§4):
`who` as its OWN field rather than glued onto the snippet, `hot` for
the blue dot, and a `preview {title, meta}` + `noun` in the SERVICE's
own shape, so the card header reads "#platform · 52 messages" rather
than one generic caption for eight different services.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from app.services import automation_verbs as verbs

from . import registry as reg

logger = logging.getLogger(__name__)

#: Connectors with a reader. Everything else answers `not_supported`
#: BY NAME — an unnamed refusal reads as a bug.
SUPPORTED = ("gmail", "outlook", "slack", "jira", "github", "teams",
             "calendar", "notion")

_MAX_GROUPS = 6
_ITEMS_PER_GROUP = 10
_MAX_ITEMS = 60
#: §5 — a picker, not a directory. Eight is what fits the popup's list
#: before it becomes a scroll of its own, and every source past it is a
#: provider row nobody reads.
_MAX_SOURCES = 8
#: One connector's children (per-channel reads, per-repo reads) run in
#: parallel but never unbounded: each is a real provider call on the
#: user's rate budget.
_FANOUT = 4
#: Per-call and whole-envelope deadlines. Both sit inside the platform
#: proxy's 30 s forwarder budget on purpose: past that the app gets a
#: 502 with no sentence, and this module's whole contract is that a
#: failure arrives NAMED. The agent must be the one that gives up.
_CALL_TIMEOUT_S = 10.0
_TOTAL_TIMEOUT_S = 22.0


# ── failure vocabulary ───────────────────────────────────────────────

def _reason_for(connector_id: str, result: dict) -> dict:
    """One dispatch envelope → the reason the user reads.

    Every branch names the account, because "could not read" with no
    subject is indistinguishable from a bug in the app.
    """
    name = verbs.display_name(connector_id) or connector_id
    kind = str((result or {}).get("kind") or "unknown")
    message = str((result or {}).get("message") or "")[:300]
    if kind == "reauth_required":
        return {
            "code": "reconnect",
            "sentence": f"{name} needs signing in again before it can "
                        f"be read.",
            "retryable": False,
            "reauth_url": (result or {}).get("reauth_url") or "",
        }
    if kind == "scope_missing":
        return {
            "code": "scope_missing",
            "sentence": f"{name} has not given enough access to read "
                        f"this.",
            "retryable": False,
            "required_scope": (result or {}).get("required_scope") or "",
        }
    if kind in ("provider_down", "rate_limited"):
        return {
            "code": "unreachable",
            "sentence": f"{name} is not answering right now. Nothing "
                        f"about your automation is wrong.",
            "retryable": True,
        }
    if kind == "tool_error" and (result or {}).get("retryable"):
        return {
            "code": "unreachable",
            "sentence": f"Could not reach {name} just now.",
            "retryable": True,
            "detail": message,
        }
    return {
        "code": "refused",
        "sentence": f"{name} refused that read.",
        "retryable": False,
        "detail": message,
    }


def _envelope(
    connector_id: str, *, focus: list, groups: Optional[list] = None,
    reason: Optional[dict] = None, truncated: bool = False,
    total: Optional[int] = None,
) -> dict:
    """The uniform envelope. `total` is the account's REAL total behind
    the preview — the unread the provider counted, not the rows we
    happen to hold — and stays None when nothing counted it. §4 makes
    that distinction load-bearing: a header reading "18 unread" over a
    number this module invented is the same class of lie as an empty
    list for an unreachable account."""
    groups = groups or []
    count = sum(len(g.get("items") or []) for g in groups)
    return {
        "account_id": connector_id,
        "connector_id": connector_id,
        "name": verbs.display_name(connector_id) or connector_id,
        "ok": reason is None,
        "reason": reason,
        "groups": groups,
        "count": count,
        "total": total,
        "truncated": bool(truncated),
        "focus": list(focus or []),
    }


def _group(key: str, label: str, kind: str, *, items=None,
           reason: Optional[dict] = None, pinned: bool = False,
           pin: Optional[dict] = None, meta: str = "",
           short: Optional[str] = None, count: Optional[int] = None) -> dict:
    """One source's worth of the account.

    `count` is the group's REAL size where the provider gave one and
    None otherwise — same rule as the envelope's `total`, for the same
    reason. `selected` is stamped by `_shape` from the picked-source
    list, never here: this function has no idea what the automation
    picked, and a default of False written at construction would be
    indistinguishable from a real "not picked".
    """
    items = list(items or [])
    return {"key": key, "label": label, "kind": kind,
            "items": items, "reason": reason,
            "pinned": bool(pinned), "pin": pin,
            "meta": meta,
            "short": _clip(short if short is not None else label, 60),
            # A group that FAILED counts nothing: 0 there would be this
            # module's word for "there is nothing here", said about an
            # account it could not read — the exact confusion the whole
            # file exists to prevent.
            "count": None if reason is not None else (
                count if count is not None else len(items))}


# ── time ─────────────────────────────────────────────────────────────

def _iso(value: Any) -> Optional[str]:
    """Whatever the provider stamped → one ISO-8601 UTC string.

    Four vendors, four formats (RFC 2822 headers, Slack's unix float,
    Jira's `+0000` offset, Graph's Z). The app formats ONE, so the
    conversion belongs here rather than in four places on a phone.
    None on anything unparseable — a wrong time reads as a real time.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(
                float(value), tz=timezone.utc,
            ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        except (ValueError, OverflowError, OSError):
            return None
    text = str(value).strip()
    # Slack: "1756400000.001200"
    try:
        if text.replace(".", "", 1).isdigit() and "." in text:
            return _iso(float(text))
    except ValueError:
        pass
    for parse in (_iso_fromisoformat, _iso_from_rfc2822):
        got = parse(text)
        if got is not None:
            return got
    return None


def _iso_fromisoformat(text: str) -> Optional[str]:
    candidate = text.replace("Z", "+00:00")
    # Jira ships "+0000" (no colon), which fromisoformat rejects.
    if len(candidate) > 5 and candidate[-5] in "+-" and ":" not in candidate[-5:]:
        candidate = candidate[:-2] + ":" + candidate[-2:]
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_from_rfc2822(text: str) -> Optional[str]:
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")


def _clip(text: Any, n: int = 160) -> str:
    s = " ".join(str(text or "").split())
    return s[:n]


def _strip_html(text: Any) -> str:
    """Teams bodies arrive as HTML (`body_content_type: 'html'`); ink on
    the phone must never carry markup."""
    return re.sub(r"<[^>]+>", " ", str(text or ""))


# ── the service's own shape (§4) ─────────────────────────────────────
#
# One preview card per account, headed in the words that service uses:
# "Inbox · 18 unread", "#platform · 52 messages", "Pull requests · 2
# open". A single generic caption over eight services is what made the
# card read as a debug dump rather than as the account.
#
# `title` is None where the LEADING GROUP's own label is the truer
# header — a Slack card names the channel it is showing, and hard-coding
# "Channels" there would title it after the picker instead.
#
# `scope` says WHAT the number counts, because the services disagree and
# guessing wrong is a lie in one word: mail counts the unread it could
# prove, Slack counts the messages in the channel on screen, Teams
# counts the chats themselves, everything else counts its rows.

_SERVICE: dict[str, dict] = {
    "gmail": {"noun": "messages", "title": "Inbox",
              "unit": "unread", "scope": "total"},
    "outlook": {"noun": "messages", "title": "Inbox",
                "unit": "unread", "scope": "total"},
    "slack": {"noun": "messages", "title": None,
              "unit": "messages", "scope": "lead"},
    "teams": {"noun": "posts", "title": "Team channels",
              "unit": "channels", "scope": "groups"},
    "jira": {"noun": "issues", "title": "Assigned to you",
             "unit": "open", "scope": "rows"},
    "github": {"noun": "pull requests", "title": "Pull requests",
               "unit": "open", "scope": "rows"},
    "notion": {"noun": "pages", "title": "Tagged to you",
               "unit": "pages", "scope": "rows"},
    "calendar": {"noun": "meetings", "title": "Your week",
                 "unit": "meetings", "scope": "rows"},
}


def _row(kind: str, *, id: Any, who: Any = "", primary: Any = "",
         secondary: Any = "", tertiary: Any = "", at: Optional[str] = None,
         hot: bool = False, pin: Optional[dict] = None,
         title: Optional[str] = None, sub: Optional[str] = None) -> dict:
    """One preview row, in the four slots the design draws (§5.5).

    `who` is its OWN field. It used to be glued onto the snippet
    ("Sara Chen — can we move it") for a single-group account and
    pushed into a third line for a multi-group one, so the app could
    not draw who/line/snippet/when at all — it had one string where the
    design has two, and no way to take it apart.

    `title`/`sub` are kept beside `primary`/`secondary` because the
    shipped app reads them: a client on the store today renders a row
    out of those two keys, and dropping them would empty every popup on
    every phone that has not updated. Same reason §2.1 keeps `output`.
    They are passed EXPLICITLY wherever the old pair did not mean
    (primary, who + secondary) — Slack titled a row with the speaker and
    Jira's sub carries its due date — because a caller that lets them be
    derived is silently rewriting what a shipped phone renders.
    """
    who = _clip(who, 80)
    primary = _clip(primary, 120)
    secondary = _clip(secondary, 160)
    return {
        "id": str(id or ""),
        "kind": kind,
        "who": who,
        "primary": primary,
        "secondary": secondary,
        "tertiary": _clip(tertiary, 120),
        "at": at,
        "hot": bool(hot),
        "pinned": False,
        "pin": pin,
        # ── older clients (R38-R42 shape) ──
        "title": _clip(title, 120) if title is not None else primary,
        "sub": _clip(sub, 200) if sub is not None else (
            f"{who} — {secondary}" if who and secondary
            else (who or secondary)),
    }


def _plural(n: int, unit: str) -> str:
    """"1 message", "52 messages".

    Only the count words this file uses, and only where the singular is
    a plain trimmed "s" — "open" and "unread" are adjectives and stay
    put. It is one character, and it is the difference between a line
    that reads as written and one that reads as generated.
    """
    if n == 1 and unit in ("messages", "channels", "meetings", "pages",
                           "databases", "posts", "issues"):
        return unit[:-1]
    return unit


def _count_meta(n: Optional[int], unit: str,
                trailer: str = "") -> str:
    """`"52 messages"`, `"4 open · 1 due Thursday"`, or `""`.

    Empty rather than "0 messages" when there is nothing: the group is
    already saying that by being empty, and a zero in the meta line
    reads as a failed count.
    """
    if not n:
        return trailer
    head = f"{n} {_plural(n, unit)}"
    return f"{head} · {trailer}" if trailer else head


def _due_word(iso: Optional[str], now: Optional[datetime] = None) -> str:
    """`"due Thursday"` / `"due today"` / `""` for a date further out.

    The design's own phrasing ("4 open · 1 due Thursday"). A weekday
    name is only unambiguous inside a week, so anything past that says
    nothing rather than naming a day the user would read as this week.
    """
    if not iso:
        return ""
    parsed = _iso_fromisoformat(iso[:19] if len(iso) > 19 else iso)
    if parsed is None:
        return ""
    now = now or datetime.now(timezone.utc)
    days = (datetime.fromisoformat(parsed.replace("Z", "+00:00")).date()
            - now.date()).days
    if days < 0:
        return "1 overdue"
    if days == 0:
        return "1 due today"
    if days == 1:
        return "1 due tomorrow"
    if days <= 6:
        weekday = datetime.fromisoformat(
            parsed.replace("Z", "+00:00")).strftime("%A")
        return f"1 due {weekday}"
    return ""


def _within(iso: Optional[str], hours: float,
            now: Optional[datetime] = None) -> bool:
    """Is that stamp inside `hours` of now, forward or back?

    The one clock question every `hot` rule asks — a meeting inside 24
    hours, a page edited today, a ticket due inside two days — so it is
    asked once and fails False on anything unparseable. A wrong dot is
    worse than no dot: it is the app claiming something is urgent.
    """
    if not iso:
        return False
    try:
        when = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return abs((when - now).total_seconds()) <= hours * 3600.0


def _due_by(iso: Optional[str], hours: float,
            now: Optional[datetime] = None) -> bool:
    """Is that DEADLINE at or inside `hours` from now — overdue
    included?

    Deliberately not `_within`: a deadline that passed three days ago is
    the most urgent row on the board, and a symmetric window drops it
    off the dot the moment it goes past the bound. "Near now" and "due
    by now" are different questions and reading one as the other is how
    an overdue ticket stops being marked.
    """
    if not iso:
        return False
    try:
        when = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (when - (now or datetime.now(timezone.utc))
            ).total_seconds() <= hours * 3600.0


# ── pins ─────────────────────────────────────────────────────────────
#
# R39. Every row and group carries `pin`: the focus entry that tapping
# "+" on it MEANS, in the `FOCUS_KINDS` vocabulary the pin endpoint
# accepts. The app sends it VERBATIM. Before this the app invented a
# kind from its icon vocabulary ('slack', 'jira', 'email'→'thread') and
# every Slack/Jira/GitHub pin was refused `bad_focus_kind` — while a
# Gmail row pinned a bare message id as a 'thread', which is the
# "(no sub…" ghost chip on the founder's canvas. `pin: null` means the
# row is information, not a place — the app draws no "+" on it.
#
# R42. A row pins ITSELF, never its container. Every item used to carry
# its group's descriptor — the channel, the sender, the project — and
# the app judges "is this row pinned?" by that descriptor's id, so one
# tap on one #all-toup message drew a checkmark on all ten of them
# while the canvas badge (a real count) said 1 (founder P4).

def _pin(kind: str, target_id: Any, label: Any) -> Optional[dict]:
    tid = str(target_id or "").strip()
    if not tid:
        return None
    return {"kind": kind, "id": tid, "label": _clip(label, 80) or tid}


#: A ROW pin's id is `<container id>#<row id>`: the row is what the user
#: tapped, and the container half is the part a scoped read can still be
#: pointed at (a channel to read, a repo to list). `container_of` is the
#: one place that format is taken apart.
_ROW_SEP = "#"


def _row_pin(kind: str, container_id: Any, row_id: Any,
             label: Any) -> Optional[dict]:
    """One row inside a container → its own pin, or None.

    Both halves are required: a row with no id of its own cannot be
    told apart from its neighbours, and a pin that cannot say WHICH row
    it means is the bug this exists to stop.
    """
    cid = str(container_id or "").strip()
    rid = str(row_id or "").strip()
    if not (cid and rid):
        return None
    return _pin(kind, f"{cid}{_ROW_SEP}{rid}", label)


def container_of(connector_id: str, pin: dict) -> Optional[tuple[str, str]]:
    """One pin → `(container id, label)`, or None if it names no place.

    The channel a message sits in, the chat a Teams message sits in, the
    repo a pull request sits in, the project a ticket belongs to — the
    thing a listing or a read can actually be aimed at. A CONTAINER pin
    is its own container and keeps its label; a ROW pin's label
    describes the row, so the caller is handed `""` and takes the name
    from the provider's own listing instead.

    Public because the pin format has to be read outside this module
    too (the executor's `_apply_focus_scope` aims a run's read at the
    same place the sheet grouped by), and a format with two readers has
    two definitions the day they drift.
    """
    if not isinstance(pin, dict):
        return None
    kind = str(pin.get("kind") or "")
    pid = str(pin.get("id") or "").strip()
    if not pid:
        return None
    label = _clip(pin.get("label") or "", 80)
    head = pid.rsplit(_ROW_SEP, 1)[0].strip() if _ROW_SEP in pid else pid
    if not head:
        return None
    own = (pid, label or pid)
    row = (head, "")
    if connector_id == "slack":
        if kind in ("channel", "thread"):
            return own if head == pid else row
    elif connector_id == "teams":
        if kind in ("thread", "channel"):
            return own if head == pid else row
    elif connector_id == "github":
        if kind == "repo":
            return own
        if kind == "ticket" and head != pid:
            return row
    elif connector_id == "notion":
        # A page and a database ARE places — you open one, you append to
        # one — so a notion pin is its own container. Neither kind is in
        # `workflow._DESTINATION_KINDS`, so this adds no write target;
        # it only lets `_read_notion` order a pinned page first.
        if kind in ("doc", "board"):
            return own if head == pid else row
    elif connector_id == "jira":
        if kind == "project":
            return own
        if kind == "ticket":
            # "ENG-12" names project "ENG". A key with no dash names no
            # project, and there is then nothing to group it under.
            proj = pid.rsplit("-", 1)[0] if "-" in pid else ""
            return (proj, proj) if proj else None
    return None


def _ordered(pinned: list, listed: list) -> tuple[list, bool]:
    """Pinned containers first, the rest of the account behind them.

    R42, founder P6: every reader used to treat a stored pin as a SCOPE
    — one pin and the sheet showed that channel and nothing else, so
    the first pin a user made was the last thing they could ever pick
    in there. A pin ORDERS this list; it never shortens it. The cap is
    `_MAX_GROUPS`, and the pinned half takes those slots first.
    """
    seen = {cid for cid, _ in pinned}
    rows = pinned + [(cid, label) for cid, label in listed if cid not in seen]
    return rows[:_MAX_GROUPS], len(rows) > _MAX_GROUPS


def _pinned_containers(
    connector_id: str, focus: list,
) -> list[tuple[str, str]]:
    """The pins of one account → their containers, in pin order, once
    each. A user who pinned two messages in #eng pinned #eng once."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for p in focus or []:
        got = container_of(connector_id, p)
        if got is None or got[0] in seen:
            continue
        seen.add(got[0])
        out.append(got)
    return out


def _from_header(value: Any) -> tuple[str, str]:
    """RFC 5322 From → (display name, address). Either may be ''."""
    from email.utils import parseaddr
    name, addr = parseaddr(str(value or ""))
    return _clip(name, 80), addr.strip().lower()


# ── dispatch ─────────────────────────────────────────────────────────

async def _call(
    user_id: str, connector_id: str, tool: str, params: dict,
) -> tuple[Optional[dict], Optional[dict]]:
    """(content, reason). Exactly one of the two is None."""
    result = await reg.dispatch_via_platform(
        user_id, connector_id=connector_id, tool_name=tool,
        tool_input=params, timeout_s=_CALL_TIMEOUT_S,
    )
    if (result or {}).get("kind") != "ok":
        return None, _reason_for(connector_id, result or {})
    try:
        content = json.loads((result or {}).get("content") or "{}")
    except (ValueError, TypeError):
        return None, {
            "code": "unreadable",
            "sentence": f"{verbs.display_name(connector_id) or connector_id} "
                        f"answered with something this could not read.",
            "retryable": True,
        }
    return (content if isinstance(content, dict) else {}), None


async def _gather(coros: list):
    """Bounded fan-out — one connector's children, `_FANOUT` at a time."""
    sem = asyncio.Semaphore(_FANOUT)

    async def _one(c):
        async with sem:
            return await c
    return await asyncio.gather(*(_one(c) for c in coros))


# ── per-connector readers ────────────────────────────────────────────

def _mail_items(messages: list, connector_id: str,
                unread: Optional[set] = None) -> list[dict]:
    """Recent mail → rows.

    `hot` MEANS UNREAD here, and each provider proves it differently:
    Graph puts `is_read` on every Outlook row, and Gmail's list call
    finally carries `labelIds` (R43 — the per-message GET it already
    makes has always returned them and they were being thrown away), so
    unread is a property of the row rather than a second read.

    `unread` is the FALLBACK, not the source: a Gmail row that carries
    no `labelIds` key at all — an older tenant, or a caller that asked
    for no body — falls back to the id set from the `is:unread` probe,
    and `unread=None` then means "nobody could say", so the row is cold
    rather than guessed hot. A row that carries the key is authoritative
    even when the list is empty; that is the difference between "read"
    and "we could not tell", and it is why this reads the key's presence
    rather than its truthiness.
    """
    items = []
    for m in messages[:_ITEMS_PER_GROUP]:
        if not isinstance(m, dict) or m.get("error"):
            continue
        mid = str(m.get("id") or "")
        if connector_id == "gmail":
            headers = m.get("headers") or {}
            raw_from = headers.get("From")
            subject = _clip(headers.get("Subject"), 120)
            at = _iso(headers.get("Date"))
            labels = m.get("labelIds")
            hot = ("UNREAD" in labels if labels is not None
                   else bool(unread) and mid in unread)
        else:
            raw_from = m.get("from")
            subject = _clip(m.get("subject"), 120)
            at = _iso(m.get("received_at"))
            hot = m.get("is_read") is False
        name, addr = _from_header(raw_from)
        sender = name or addr or _clip(raw_from, 80)
        snippet = _clip(m.get("snippet") or m.get("preview"), 140)
        # Gmail hands the conversation id back with every message;
        # Graph's `$select` on the list call does not, so an Outlook row
        # pins the message it is. Either way the row means ONE thread —
        # pinning a mail from Sara must not tick her other five.
        thread_id = str(m.get("threadId") or "").strip() or mid
        items.append(_row(
            "message", id=mid, who=sender,
            primary=subject or "(no subject)", secondary=snippet,
            at=at, hot=hot,
            pin=_pin("thread", thread_id, subject or sender),
        ))
    return items


async def _read_mail(user_id: str, connector_id: str, focus: list) -> dict:
    """gmail / outlook — recent messages: sender, subject, snippet.

    A pinned person or label becomes its OWN group with its own query,
    because that is the question the pin was made to ask ("what is in
    here from Sara?"), not a filter applied to a list of everything.
    The plain recent read runs ALONGSIDE those and never instead of
    them: the sender pin lives on the group it makes, so the group is
    where a whole correspondent is pinned and the rows below it are
    each their own conversation.
    """
    tool = f"{connector_id}__list_messages"
    # key, label, params, pinned, group pin
    queries: list[tuple[str, str, dict, bool, Optional[dict]]] = []
    for pin in focus:
        kind, pid = pin.get("kind"), str(pin.get("id") or "").strip()
        if not pid:
            continue
        label = str(pin.get("label") or pid)
        if kind == "person":
            queries.append((f"from:{pid}", f"From {label}",
                            {"query": f"from:{pid}"},
                            True, _pin("person", pid, label)))
        elif kind in ("label", "folder"):
            # R43 — an Outlook folder is a COLLECTION, not a search
            # term. `outlook__list_messages` takes `folder` and scopes
            # the read to it; the id used to go out as `$search`, i.e.
            # the whole mailbox was read and the group matched nothing.
            queries.append((
                f"label:{pid}", label,
                {"query": f"label:{pid}"} if connector_id == "gmail"
                else {"folder": pid},
                True, _pin(kind, pid, label)))
    truncated = len(queries) > _MAX_GROUPS - 1
    queries = queries[:_MAX_GROUPS - 1]
    queries.append(("recent", "Recent", {}, False, None))

    # include_body means a DIFFERENT thing per provider: outlook's False
    # still $selects subject/from/preview, gmail's False strips the list
    # to bare {id, threadId} — which is how every Gmail row rendered as
    # "(no subject)" with no sender and no time (R39). Gmail needs True
    # to get headers + snippet at all; ten parallel fetches sit well
    # inside _CALL_TIMEOUT_S.
    calls = [
        _call(user_id, connector_id, tool,
              {"max_results": _ITEMS_PER_GROUP,
               "include_body": connector_id == "gmail",
               **params})
        for _k, _l, params, _p, _g in queries
    ]
    # R43 — the blue dot is per-row now (`labelIds`, above). This probe
    # SHRINKS rather than vanishing, because its second product cannot
    # come from anywhere else: `total` is §4's "the account's REAL total
    # behind the preview", and a ten-row page cannot supply it. One id
    # is enough to read `result_size` off. The ids it does return stay
    # as `_mail_items`' fallback for a row that carries no `labelIds`.
    # Outlook needs none of it: `is_read` is already on every row.
    probe = connector_id == "gmail"
    if probe:
        calls.append(_call(user_id, connector_id, tool, {
            "max_results": 1, "include_body": False,
            "query": "is:unread in:inbox",
        }))
    results = await _gather(calls)
    unread: Optional[set] = None
    total: Optional[int] = None
    if probe:
        content, reason = results.pop()
        if reason is None:
            unread = {str(m.get("id") or "")
                      for m in (content.get("messages") or [])
                      if isinstance(m, dict)}
            size = content.get("result_size")
            # NOT `len(unread)`: the probe asks for one id, so counting
            # what came back would report "1 unread" for a mailbox
            # holding two hundred. No count is better than a wrong one.
            total = int(size) if isinstance(size, int) else None
    groups, hard_reason = [], None
    for (key, label, _params, pinned, gpin), (content, reason) in zip(
            queries, results):
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(key, label, "mailbox", reason=reason,
                                 pinned=pinned, pin=gpin))
            continue
        items = _mail_items(content.get("messages") or [], connector_id,
                            unread)
        hot = sum(1 for i in items if i["hot"])
        groups.append(_group(
            key, label, "mailbox", pinned=pinned, pin=gpin, items=items,
            meta=_count_meta(len(items), "messages",
                             f"{hot} unread" if hot else ""),
        ))
    if total is None and connector_id == "outlook":
        # Graph is never asked for a mailbox count here, so the honest
        # total is the unread this read could actually see.
        seen = sum(1 for g in groups for i in g["items"] if i["hot"])
        total = seen or None
    # Every group failed for the same reason ⇒ the ACCOUNT could not be
    # read, and the envelope says so rather than showing N empty rows.
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated, total=total)


async def _read_slack(user_id: str, connector_id: str, focus: list) -> dict:
    """slack — recent messages per JOINED conversation, pinned first.

    `is_member` is the filter, exactly as `automations__list_targets`
    filters it (R38): the manifest has no `chat:write.public`, so a
    channel the workspace never joined is neither readable nor
    postable, and offering it is how #general got named.

    The listing runs even when there are pins. A pinned channel that
    the listing cannot confirm is still read — the user pinned it, and
    a workspace that will not list its channels is not a reason to
    forget where this automation already starts.
    """
    pinned = _pinned_containers(connector_id, focus)
    content, reason = await _call(
        user_id, connector_id, "slack__list_channels",
        {"types": "public_channel,private_channel", "limit": 100},
    )
    if reason is not None and not pinned:
        return _envelope(connector_id, focus=focus, reason=reason)
    listed = [
        (str(c["id"]), f"#{c.get('name')}" if c.get("name")
         else str(c.get("user_name") or c["id"]))
        for c in ((content or {}).get("channels") or [])
        if isinstance(c, dict) and c.get("is_member") and c.get("id")
    ]
    names = dict(listed)
    rows, truncated = _ordered(
        [(cid, names.get(cid) or label or cid) for cid, label in pinned],
        listed,
    )
    if not rows:
        # Joined nothing is a real, readable answer — no groups, not a
        # reason. The user has a Slack; they are in no channel this
        # automation could start from.
        return _envelope(connector_id, focus=focus, groups=[])
    is_pinned = {cid for cid, _ in pinned}

    results = await _gather([
        _call(user_id, connector_id, "slack__read_messages",
              {"channel": cid, "limit": _ITEMS_PER_GROUP})
        for cid, _label in rows
    ])
    groups, hard_reason = [], None
    for (cid, label), (content, reason) in zip(rows, results):
        chan_pin = _pin("channel", cid, label)
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(cid, label, "channel", reason=reason,
                                 pinned=cid in is_pinned, pin=chan_pin))
            continue
        items = []
        for m in (content.get("messages") or [])[:_ITEMS_PER_GROUP]:
            if not isinstance(m, dict):
                continue
            ts = str(m.get("ts") or "")
            # A reply carries `in_thread_of`; a parent that HAS replies
            # carries `thread_ts` (its own ts). Either way the row means
            # one conversation, so two replies under one message are one
            # pin and two separate messages are two.
            root = str(m.get("thread_ts") or m.get("in_thread_of") or ts)
            who = _clip(m.get("from"), 80) or "(app)"
            text = _clip(m.get("text"), 160)
            items.append(_row(
                "message", id=ts, who=who, primary=text or who,
                at=_iso(ts), title=who, sub=text,
                # Slack's Web API hands a user token no per-message read
                # state, so "unread" is not answerable here. A message
                # that NAMES you is — the provider marks it, from the
                # same `<@U…>` run it rewrites into a display name — and
                # "someone is asking you" is what the dot is for.
                hot=bool(m.get("mentions_me")),
                pin=_row_pin("thread", cid, root,
                             _clip(f"{who}: {text}", 60) if text else who),
            ))
        named = sum(1 for i in items if i["hot"])
        groups.append(_group(
            cid, label, "channel", items=items,
            pinned=cid in is_pinned, pin=chan_pin,
            meta=_count_meta(len(items), "messages",
                             f"{named} name you" if named else ""),
        ))
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated)


#: A Jira project key as the provider mints them (letters, digits and
#: `_`, opening on a letter). A pin whose id is not one names no project
#: a JQL can be aimed at, so it gets no scoped read and no group.
_JIRA_KEY = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,60}$")
#: The two halves of every read below. The ORDER BY has to trail the
#: WHOLE query — JQL forbids it inside parentheses — which is why a
#: scope is a query of its own rather than a prefix on this one.
_JIRA_MINE = "assignee = currentUser() AND resolution = Unresolved"
_JIRA_ORDER = "ORDER BY duedate ASC, updated DESC"


def _jira_item(issue: Any) -> Optional[tuple[str, dict]]:
    """One issue → `(project key, row)`, or None if it is not an issue."""
    if not isinstance(issue, dict):
        return None
    key = str(issue.get("key") or "")
    due = _iso(issue.get("duedate"))
    # The provider ships the project key; a key like "ENG-12" names it
    # too, which is the only route left when it does not.
    proj = str(issue.get("project") or "").strip() or (
        key.rsplit("-", 1)[0] if "-" in key else "")
    status = _clip(issue.get("status") or "", 60)
    sub = status
    # The due date is the whole reason this list is ordered the way it
    # is, so it is stated rather than left as a bare timestamp the app
    # has to guess the meaning of.
    if due:
        sub = f"{sub} · due {due[:10]}" if sub else f"due {due[:10]}"
    priority = _clip(issue.get("priority") or "", 40)
    title = (f"{key} · {_clip(issue.get('summary'), 110)}" if key
             else _clip(issue.get("summary"), 110))
    assignee = issue.get("assignee")
    if isinstance(assignee, dict):
        assignee = assignee.get("display_name") or assignee.get("name")
    return proj, _row(
        "ticket", id=key,
        # A ticket's "who" is who it is ON — the assignee — not who
        # filed it. This read is `assignee = currentUser()`, so it is
        # the user themselves and the app draws the slot as such.
        who=assignee or "",
        primary=title, secondary=status,
        tertiary=f"due {due[:10]}" if due else "",
        at=due or _iso(issue.get("updated")),
        # Hot is what breaks first: a P1/P2, or a due date inside two
        # days. Both are the whole reason the list is ordered by
        # duedate, so the dot marks the rows that ordering is FOR.
        hot=(priority.lower() in ("highest", "high")
             or _due_by(due, 48)),
        pin=_pin("ticket", key, key),
        title=title, sub=sub,
    )


async def _read_jira(user_id: str, connector_id: str, focus: list) -> dict:
    """jira — the user's own open tickets by project, soonest due first.

    A pinned project is read by a query OF ITS OWN, alongside the
    unscoped one, exactly as a pinned sender is in `_read_mail`. R42
    took the `project in (…)` prefix off the unscoped read — it made
    every other project unpickable the moment one pin existed — but
    bucketing that one global window per project left a pinned board
    rendering EMPTY while it held open tickets, and `items: []` with no
    reason is this module's word for "there is nothing here". The scope
    now lives where it filters nothing the other groups can see.
    """
    pins = [cid for cid, _ in _pinned_containers(connector_id, focus)
            if _JIRA_KEY.match(cid)]
    # One group slot is left for the account's own projects, and every
    # scoped query is a real call on the user's rate budget.
    truncated = len(pins) > _MAX_GROUPS - 1
    pins = pins[:_MAX_GROUPS - 1]
    pinned = set(pins)
    fields = "summary,status,duedate,priority,updated,project"
    # The unscoped read leads, and it stays unscoped: every group below
    # the pinned ones is built out of what it returns, so it is asked
    # for enough issues to fill them.
    reads = [(None, f"{_JIRA_MINE} {_JIRA_ORDER}", _ITEMS_PER_GROUP * 2)]
    reads += [(p, f'project = "{p}" AND {_JIRA_MINE} {_JIRA_ORDER}',
               _ITEMS_PER_GROUP) for p in pins]
    results = await _gather([
        _call(user_id, connector_id, "jira__search_issues",
              {"jql": jql, "max_results": limit, "fields": fields})
        for _p, jql, limit in reads
    ])

    by_project: dict[str, list] = {p: [] for p in pins}
    group_reason: dict[str, dict] = {}
    (all_content, hard_reason), scoped = results[0], results[1:]
    for (proj, _jql, _limit), (content, reason) in zip(reads[1:], scoped):
        if reason is not None:
            group_reason[proj] = reason
            continue
        for issue in ((content or {}).get("issues") or []):
            got = _jira_item(issue)
            if got is not None:
                by_project[proj].append(got[1])
    for issue in ((all_content or {}).get("issues") or []):
        got = _jira_item(issue)
        if got is None:
            continue
        proj, item = got
        # A pinned project has its own read; the copy in this window
        # would be the same rows a second time, out of that read's order.
        if proj in pinned:
            continue
        by_project.setdefault(proj, []).append(item)

    order = pins + [p for p in by_project if p not in pinned]
    truncated = truncated or len(order) > _MAX_GROUPS or any(
        len(v) > _ITEMS_PER_GROUP for v in by_project.values())
    groups = []
    for proj in order[:_MAX_GROUPS]:
        items = (by_project.get(proj) or [])[:_ITEMS_PER_GROUP]
        meta = _count_meta(len(items), "open", _soonest_due(items))
        if proj:
            groups.append(_group(proj, proj, "tickets", items=items,
                                 reason=group_reason.get(proj),
                                 pinned=proj in pinned, meta=meta,
                                 pin=_pin("project", proj, proj)))
        else:
            # A key with no dash and no project field names no project:
            # there is nothing to scope a JQL to, so no group pin.
            groups.append(_group("other", "Assigned to you", "tickets",
                                 items=items, meta=meta))
    # Every read failed ⇒ the ACCOUNT could not be read, and the
    # envelope says so rather than showing N empty projects.
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    if not groups:
        groups = [_group("other", "Assigned to you", "tickets", items=[])]
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated)


def _soonest_due(items: list) -> str:
    """"1 due Thursday" for the nearest deadline in a group, or "".

    The design's own meta line ("4 open · 1 due Thursday"). It names ONE
    date because a list of them is not a meta line, and the soonest is
    the one that decides whether the group is opened now.
    """
    dues = sorted(i["tertiary"][4:] for i in items
                  if i.get("tertiary", "").startswith("due "))
    return _due_word(dues[0]) if dues else ""


async def _read_github(user_id: str, connector_id: str, focus: list) -> dict:
    """github — open pull requests, per repository, pinned repos first.

    `github__list_issues` is the only listing tool the manifest has and
    it returns issues AND pull requests, flagged; the PRs are filtered
    out HERE rather than asked for, because there is no tool that asks.
    """
    pinned = _pinned_containers(connector_id, focus)
    content, reason = await _call(
        user_id, connector_id, "github__list_repos",
        {"sort": "pushed", "per_page": 30},
    )
    if reason is not None and not pinned:
        return _envelope(connector_id, focus=focus, reason=reason)
    # Three unpinned repos, not six: every one of them is a second
    # provider call on the user's rate budget, and the pinned ones are
    # the answer the sheet exists to give.
    listed = [(str(r.get("full_name")), str(r.get("full_name")))
              for r in ((content or {}).get("repos") or [])
              if isinstance(r, dict) and r.get("full_name")][:3]
    rows, truncated = _ordered(
        [(cid, label or cid) for cid, label in pinned], listed)
    is_pinned = {cid for cid, _ in pinned}
    pairs = [(key, key.split("/", 1)) for key, _label in rows if "/" in key]
    if not pairs:
        # A pin that names no owner half is not a repo this can read, so
        # a failed listing is still the only answer there is.
        if reason is not None:
            return _envelope(connector_id, focus=focus, reason=reason)
        return _envelope(connector_id, focus=focus, groups=[])
    results = await _gather([
        _call(user_id, connector_id, "github__list_issues",
              {"owner": owner, "repo": repo, "state": "open", "per_page": 30})
        for _key, (owner, repo) in pairs
    ])
    groups, hard_reason = [], None
    for (key, _pair), (content, reason) in zip(pairs, results):
        repo_pin = _pin("repo", key, key)
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(key, key, "repo", reason=reason,
                                 pinned=key in is_pinned, pin=repo_pin))
            continue
        items = []
        for i in (content.get("issues") or []):
            if not isinstance(i, dict) or not i.get("is_pull_request"):
                continue
            number = str(i.get("number") or "")
            title = _clip(i.get("title"), 120)
            author = _clip(i.get("user") or "", 80)
            items.append(_row(
                "pull_request", id=number, who=author, primary=title,
                secondary=f"#{number}",
                at=_iso(i.get("updated_at")),
                # `/repos/{o}/{r}/issues` carries no check status, so
                # "failing checks" cannot be answered from this read.
                # What it DOES carry is `draft`, and a pull request that
                # is not a draft is the one waiting on a human — which
                # is the same question the dot is asked.
                #
                # R43 shipped the two tools that COULD answer it
                # literally — `github__list_check_runs` (one commit's
                # conclusions) and `github__search_issues`
                # (`status:failure` across repositories) — but neither
                # is this read: turning the dot red here would cost one
                # extra call per pull request on the user's rate budget,
                # for a group the sheet lists by repository. The chip
                # ("Failing checks first") and the `build_red` event are
                # where those tools earn their place.
                hot=i.get("draft") is False,
                pin=_row_pin("ticket", key, number,
                             _clip(f"#{number} {title}", 60)),
                title=title,
                sub=f"#{number} · {i.get('user') or ''}".strip(" ·"),
            ))
        items = items[:_ITEMS_PER_GROUP]
        waiting = sum(1 for i in items if i["hot"])
        groups.append(_group(
            key, key, "repo", items=items,
            pinned=key in is_pinned, pin=repo_pin,
            short=key.split("/", 1)[-1],
            meta=_count_meta(len(items), "open",
                             f"{waiting} ready for review" if waiting else ""),
        ))
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated)


def _chat_label(c: dict) -> str:
    """A Teams chat's name: its topic, else who is in it. Module level
    because the sources list and the reader must name the same chat the
    same way — two spellings of one chat read as two chats."""
    if c.get("topic"):
        return _clip(c["topic"], 60)
    names = [m.get("display_name") or m.get("displayName") or ""
             for m in (c.get("members") or []) if isinstance(m, dict)]
    names = [n for n in names if n]
    return _clip(", ".join(names[:3]), 60) or "Chat"


async def _read_teams(user_id: str, connector_id: str, focus: list) -> dict:
    """teams — recent messages per chat, pinned chats first.

    R39: Teams was not in SUPPORTED at all, so the ONE connector the
    Morning work brief actually reads a chat from answered "There is no
    way to look inside Teams yet." — with an expired credential
    underneath that the sheet therefore never surfaced either.
    """
    pinned = _pinned_containers(connector_id, focus)
    content, reason = await _call(
        user_id, connector_id, "teams__list_chats", {"max_results": 25},
    )
    if reason is not None and not pinned:
        return _envelope(connector_id, focus=focus, reason=reason)

    chats = [c for c in ((content or {}).get("chats") or [])
             if isinstance(c, dict) and c.get("id")]
    listed = [(str(c["id"]), _chat_label(c)) for c in chats]
    labels = dict(listed)
    read_marks = {str(c["id"]): _iso(c.get("last_read_at")) for c in chats
                  if c.get("last_read_at")}
    rows, truncated = _ordered(
        [(cid, labels.get(cid) or label or cid) for cid, label in pinned],
        listed,
    )
    if not rows:
        return _envelope(connector_id, focus=focus, groups=[])
    is_pinned = {cid for cid, _ in pinned}

    results = await _gather([
        _call(user_id, connector_id, "teams__read_chat_messages",
              {"chat_id": cid, "max_results": _ITEMS_PER_GROUP})
        for cid, _label in rows
    ])
    groups, hard_reason = [], None
    for (cid, label), (content, reason) in zip(rows, results):
        chat_pin = _pin("thread", cid, label)
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(cid, label, "channel", reason=reason,
                                 pinned=cid in is_pinned, pin=chat_pin))
            continue
        items = []
        # §4's dot for Teams is genuinely unread, and Graph gives it
        # away for free: every /chats row carries the caller's own
        # `viewpoint.lastMessageReadDateTime`, so a message stamped
        # after it is one this user has not seen. None means Graph did
        # not say, and nothing is then marked.
        last_read = read_marks.get(cid)
        for m in (content.get("messages") or [])[:_ITEMS_PER_GROUP]:
            if not isinstance(m, dict) or m.get("deleted_at"):
                continue
            body = m.get("body") or ""
            if (m.get("body_content_type") or "").lower() == "html":
                body = _strip_html(body)
            who = _clip(m.get("sender"), 80) or "(system)"
            body = _clip(body, 160)
            at = _iso(m.get("created_at"))
            # A chat message has no thread of its own — Graph's chat
            # messages are flat — so the row IS the message.
            items.append(_row(
                "message", id=str(m.get("id") or ""), who=who,
                primary=body or who, at=at,
                hot=bool(last_read and at and at > last_read),
                pin=_row_pin("thread", cid, m.get("id"),
                             _clip(f"{who}: {body}", 60) if body else who),
                title=who, sub=body,
            ))
        unseen = sum(1 for i in items if i["hot"])
        groups.append(_group(
            cid, label, "channel", items=items,
            pinned=cid in is_pinned, pin=chat_pin,
            meta=_count_meta(len(items), "messages",
                             f"{unseen} unread" if unseen else ""),
        ))
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated)


async def _read_calendar(user_id: str, connector_id: str, focus: list) -> dict:
    """calendar — what is coming up, soonest first. Events are moments,
    not places, so the rows carry no pin."""
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    content, reason = await _call(
        user_id, connector_id, "calendar__list_events",
        {"max_results": _ITEMS_PER_GROUP,
         "time_min": now.replace(microsecond=0).isoformat(),
         "time_max": (now + timedelta(days=7)).replace(
             microsecond=0).isoformat()},
    )
    if reason is not None:
        return _envelope(connector_id, focus=focus, reason=reason)
    items = []
    for ev in (content.get("events") or []):
        if not isinstance(ev, dict):
            continue
        start = ev.get("start") or {}
        at = _iso(start.get("dateTime") or start.get("date"))
        bits = []
        if ev.get("location"):
            bits.append(_clip(ev["location"], 60))
        n = ev.get("attendee_count")
        if isinstance(n, int) and n > 1:
            bits.append(f"{n} people")
        items.append(_row(
            "event", id=str(ev.get("id") or ""),
            # A meeting has no "who" — §4 says the slot is then "" and
            # the app draws the row without it, never the word "None".
            # `attendees` is in this tool's `output_redaction` anyway.
            primary=_clip(ev.get("summary"), 120) or "(untitled)",
            secondary=" · ".join(bits),
            at=at,
            # The dot is "this is nearly here": a meeting inside 24
            # hours is the one a brief has to mention today.
            hot=_within(at, 24, now),
            pin=None,
            sub=" · ".join(bits),
        ))
    soon = sum(1 for i in items if i["hot"])
    return _envelope(connector_id, focus=focus, total=len(items), groups=[
        _group("upcoming", "Coming up", "events", items=items,
               short="Your week",
               meta=_count_meta(len(items), "meetings",
                                f"{soon} in the next day" if soon else "")),
    ])


#: Notion search hands back `object: "page" | "data_source"`; the FOCUS
#: kind for each is what a pin on it has to be, and picking one from
#: this sheet writes exactly that.
_NOTION_KIND = {"page": "doc", "data_source": "board"}


async def _read_notion(user_id: str, connector_id: str, focus: list) -> dict:
    """notion — what has been edited lately, pages and databases apart.

    R43: Notion was not in `_READERS` at all, so a connector the design
    puts in the PLANS band answered "There is no way to look inside
    Notion yet" — the same shape of refusal Teams wore in R39, and with
    the same consequence, that an expired credential underneath it was
    never surfaced either.

    ONE call. `notion__search` with no query is the integration's whole
    shared surface, newest edit first, and a page's body is a second
    request per row that nothing on this card would render.
    """
    pinned = [cid for cid, _ in _pinned_containers(connector_id, focus)]
    content, reason = await _call(
        user_id, connector_id, "notion__search",
        {"page_size": _ITEMS_PER_GROUP * 2, "sort": "last_edited_desc"},
    )
    if reason is not None:
        return _envelope(connector_id, focus=focus, reason=reason)
    order = {pid: n for n, pid in enumerate(pinned)}
    rows: dict[str, list] = {"doc": [], "board": []}
    for obj in (content.get("results") or []):
        if not isinstance(obj, dict):
            continue
        kind = _NOTION_KIND.get(str(obj.get("object") or ""))
        oid = str(obj.get("id") or "")
        if kind is None or not oid:
            continue
        edited = _iso(obj.get("last_edited_time"))
        title = _clip(obj.get("title"), 120) or "(untitled)"
        rows[kind].append(_row(
            # A page has no author on the search result — Notion returns
            # `last_edited_by` as a bare user id, and resolving it is a
            # request per row — so §4's "no who" case applies and the
            # slot is "".
            "page" if kind == "doc" else "database", id=oid,
            primary=title,
            secondary="database" if kind == "board" else "",
            at=edited,
            # Edited today. A page nobody has touched this week is not
            # something the brief has to raise.
            hot=_within(edited, 24),
            pin=_pin(kind, oid, title),
            sub="database" if kind == "board" else "",
        ))
    # A pinned page leads its own group, for the same reason a pinned
    # channel leads Slack's list: the pin says where this automation
    # starts, and burying it under today's edits hides that.
    for bucket in rows.values():
        bucket.sort(key=lambda r: order.get(
            (r["pin"] or {}).get("id", ""), len(order)))
    groups = []
    for kind, key, label in (("doc", "pages", "Pages"),
                             ("board", "databases", "Databases")):
        items = rows[kind][:_ITEMS_PER_GROUP]
        if not items and groups:
            continue
        hot = sum(1 for i in items if i["hot"])
        groups.append(_group(
            key, label, key, items=items,
            pinned=any((i["pin"] or {}).get("id") in order for i in items),
            meta=_count_meta(len(items), "pages" if kind == "doc" else
                             "databases",
                             f"{hot} edited today" if hot else ""),
        ))
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=len(rows["doc"]) > _ITEMS_PER_GROUP)


_READERS = {
    "gmail": _read_mail,
    "outlook": _read_mail,
    "slack": _read_slack,
    "jira": _read_jira,
    "github": _read_github,
    "teams": _read_teams,
    "calendar": _read_calendar,
    "notion": _read_notion,
}


# ── the card's own header (§4) ───────────────────────────────────────

def _shape(env: dict, *, sources: Optional[list] = None) -> dict:
    """Stamp `noun`, `preview`, `rows`, `pinned` and `selected`.

    Runs on EVERY envelope, including the ones that carry only a
    reason: §4 says each of these keys is always served, and an app
    that has to branch on "is this key here" before it can branch on
    `ok` has two failure vocabularies where the module promises one.

    It runs LAST, after truncation, because `rows` and `items` are the
    same list and a projection taken before the cap would keep the rows
    the cap just dropped.
    """
    connector_id = env["connector_id"]
    svc = _SERVICE.get(connector_id) or {
        "noun": "items", "title": None, "unit": "items", "scope": "rows"}
    groups = env.get("groups") or []
    # A pin's id is the row's own descriptor, so a row knows whether it
    # is pinned by its OWN id — never by its container's. R42's founder
    # P4 was exactly that confusion drawn as ten checkmarks.
    pinned_ids = {str(p.get("id") or "") for p in (env.get("focus") or [])
                  if isinstance(p, dict)}
    picked = {str(x) for x in (sources or []) if isinstance(x, str)}
    rows = 0
    for g in groups:
        items = g.get("items") or []
        for row in items:
            row["pinned"] = bool(row.get("pin")) and (
                str((row["pin"] or {}).get("id") or "") in pinned_ids)
        # `rows` is §4's name for this list and `items` is R38's. They
        # are the SAME objects: an older phone reads one, a current one
        # reads the other, and neither can drift from the other.
        g["rows"] = items
        # `selected` is "did the automation PICK this source", which is
        # not "did the user pin something in it" — the two are separate
        # writes with separate meanings, and merging them would tick a
        # source nobody chose.
        g["selected"] = g.get("key") in picked or (
            bool(g.get("pin")) and str((g["pin"] or {}).get("id") or "")
            in picked)
        rows += len(items)

    lead = next((g for g in groups if g.get("items")), None) or (
        groups[0] if groups else None)
    unit = svc["unit"]
    if svc["scope"] == "groups":
        n: Optional[int] = len(groups) or None
    elif svc["scope"] == "lead":
        n = len(lead.get("items") or []) if lead else None
    elif svc["scope"] == "total":
        n = env.get("total")
        if n is None:
            # The count the unit NAMES could not be taken, so the unit
            # goes with it. Falling back to the rows in hand under the
            # word "unread" would state a number of unread mails that
            # this read never established — the header saying something
            # the body cannot support, which is the §4 failure this
            # whole block is arranged to avoid.
            n, unit = rows or None, svc["noun"]
    else:
        n = rows or None
    title = svc["title"] or (lead.get("label") if lead else None) or (
        env.get("name") or connector_id)
    env["noun"] = svc["noun"]
    env["preview"] = {
        "title": _clip(title, 60),
        # No count ⇒ no meta. "0 messages" over a card that failed to
        # read is the same lie as an empty list for an unreachable
        # account, said in the header instead of the body.
        "meta": f"{n} {_plural(n, unit)}" if n else "",
    }
    return env


# ── the entry point ──────────────────────────────────────────────────

async def account_contents(
    user_id: str, *, connector_id: str, focus: Optional[list] = None,
    connection: Optional[dict] = None,
    sources: Optional[list] = None,
) -> dict:
    """What is inside one account, in the uniform envelope.

    `connection` is this user's connection-state row for the connector
    (the caller already has it); when it says the account is not usable
    we answer with THAT reason rather than making a call we know will
    fail — a "reconnect" sentence beats a provider's 401 relayed
    through three layers.

    `sources` is the automation's PICKED source ids for this account
    (§2.2). None means the caller did not say — every group then reads
    `selected: false`, which is what an unwired caller should look
    like rather than a guess.
    """
    focus = list(focus or [])
    name = verbs.display_name(connector_id) or connector_id
    # Connection state FIRST: an expired credential is the truer answer
    # than "no way to look inside" for a connector with no reader, and
    # the one with a Reconnect action on it (R39 — Teams wore its
    # expiry only as a canvas ring).
    if connection is not None:
        status = str(connection.get("status") or "")
        if not connection.get("connected"):
            return _shape(_envelope(connector_id, focus=focus, reason={
                "code": "not_connected",
                "sentence": f"{name} is not connected.",
                "retryable": False,
                "consent_url": f"/api/oauth/connect/{connector_id}",
            }), sources=sources)
        if status and status != "active":
            return _shape(_envelope(connector_id, focus=focus, reason={
                "code": "reconnect",
                "sentence": f"{name} needs signing in again before it "
                            f"can be read.",
                "retryable": False,
                "consent_url": f"/api/oauth/connect/{connector_id}",
            }), sources=sources)
    reader = _READERS.get(connector_id)
    if reader is None:
        return _shape(_envelope(connector_id, focus=focus, reason={
            "code": "not_supported",
            "sentence": f"There is no way to look inside {name} yet.",
            "retryable": False,
        }), sources=sources)
    try:
        env = await asyncio.wait_for(
            reader(user_id, connector_id, focus), _TOTAL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[automations] contents timed out for %s", connector_id)
        return _shape(_envelope(connector_id, focus=focus, reason={
            "code": "unreachable",
            "sentence": f"{name} took too long to answer. Try again.",
            "retryable": True,
        }), sources=sources)
    except Exception as e:  # noqa: BLE001 — the reason IS the answer
        logger.warning("[automations] contents failed for %s: %s: %s",
                       connector_id, type(e).__name__, str(e)[:200])
        return _shape(_envelope(connector_id, focus=focus, reason={
            "code": "unreachable",
            "sentence": f"Could not look inside {name} just now.",
            "retryable": True,
        }), sources=sources)
    if env["count"] > _MAX_ITEMS:
        kept, total = [], 0
        for g in env["groups"]:
            room = max(0, _MAX_ITEMS - total)
            g = dict(g)
            g["items"] = (g.get("items") or [])[:room]
            total += len(g["items"])
            kept.append(g)
        env["groups"] = kept
        env["count"] = total
        env["truncated"] = True
    return _shape(env, sources=sources)


# ── the account's own objects (§5) ───────────────────────────────────
#
# "What it may open here" used to be a seed table in the design deck: a
# fixed list of plausible-looking channels and labels that existed on
# nobody's account. This is the account itself — the channels you are
# in, the chats that have moved, the projects you have tickets on, the
# repositories you push to.
#
# THREE RULES, and each one is load-bearing.
#
# 1. `kind` comes from `spec.FOCUS_KINDS` and nowhere else. Picking a
#    source becomes a pin descriptor, and R39 proved an invented kind
#    409s every write with `bad_focus_kind` — silently, because the
#    canvas had already drawn the tick.
#
# 2. ONE round of provider calls, and never more than one per account.
#    §2.2 puts `sources_available` on every entry of the workflow
#    payload, which is built for the WHOLE canvas at once: a fan-out per
#    source would multiply eight accounts into forty provider calls
#    every time a user opens it, and R40's "Couldn't load automations"
#    was a 22.7 s answer abandoned at 15 s. Where one call cannot supply
#    a count, `count` is None and the meta says something true from the
#    listing rather than a number this module invented.
#
# 3. A slow or refusing provider yields `[]`, never an exception. That
#    is a real cost — this is the one list in the file where "absent"
#    and "empty" collapse — so the deadline is `_CALL_TIMEOUT_S` rather
#    than the envelope's, and the caller must treat an empty sources
#    list as "not answered", not as "this account holds nothing".

def _source(sid: Any, name: Any, meta: str, kind: str, *,
            short: Optional[str] = None,
            count: Optional[int] = None) -> Optional[dict]:
    """One real object → the §5 row, or None when it has no id.

    An id-less source cannot be picked, cannot be pinned and cannot be
    told from its neighbour, so it is dropped rather than shipped as a
    row that does nothing when tapped.
    """
    sid = str(sid or "").strip()
    if not sid:
        return None
    label = _clip(name, 80) or sid
    return {
        "id": sid,
        "name": label,
        "meta": meta,
        "short": _clip(short if short is not None else label, 40),
        "kind": kind,
        "count": count,
    }


def _ago(iso: Optional[str], now: Optional[datetime] = None) -> str:
    """"today" / "3 days ago" / "" — the meta line's tail.

    Only three resolutions, because a source row is read at a glance and
    "2 days, 14 hours" is not read at all.
    """
    if not iso:
        return ""
    try:
        when = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return ""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    days = ((now or datetime.now(timezone.utc)) - when).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days <= 30:
        return f"{days} days ago"
    return ""


#: Gmail's system labels exist on EVERY account and are the three the
#: design names. There is no `gmail__list_labels` tool, so the user's
#: own labels cannot be enumerated yet — these are real, pickable and
#: readable (`_read_mail` turns a `label` pin into `label:<id>`), which
#: is more than an empty list would be.
_GMAIL_LABELS = (
    ("INBOX", "Inbox", "everything that lands here", "label"),
    ("IMPORTANT", "Important", "what Gmail marked for you", "label"),
    ("STARRED", "Starred", "kept for later", "label"),
)


async def _sources_gmail(user_id: str, connector_id: str) -> list[dict]:
    content, reason = await _call(
        user_id, connector_id, "gmail__list_messages",
        {"max_results": 1, "include_body": False,
         "query": "is:unread in:inbox"},
    )
    size = (content or {}).get("result_size") if reason is None else None
    unread = int(size) if isinstance(size, int) else None
    out = []
    for sid, name, meta, kind in _GMAIL_LABELS:
        n = unread if sid == "INBOX" else None
        out.append(_source(sid, name,
                           f"{n} unread" if n else meta, kind, count=n))
    return out


#: §5 — "Inbox, one per mail folder (≤6), Sent, for context".
#: `outlook__list_folders` (R43) puts Inbox / Archive / Sent Items first
#: and tags them `well_known` BY ID, so the three the design names are
#: matched on the segment rather than on a display name that is
#: localised in half the world's mailboxes.
_OUTLOOK_USER_FOLDERS = 6

_OUTLOOK_FOLDER_NAMES = {
    "inbox": ("Inbox", "everything that lands here"),
    "archive": ("Archive", "what you have filed"),
    "sentitems": ("Sent, for context", "what you sent, for context"),
}


async def _sources_outlook(user_id: str, connector_id: str) -> list[dict]:
    content, reason = await _call(
        user_id, connector_id, "outlook__list_folders", {"max_results": 50},
    )
    out = []
    if reason is None:
        for f in ((content or {}).get("folders") or []):
            if not isinstance(f, dict):
                continue
            wk = str(f.get("well_known") or "")
            name, fallback = _OUTLOOK_FOLDER_NAMES.get(
                wk, (str(f.get("name") or ""), "one of your folders"))
            unread = f.get("unread_count")
            n = unread if isinstance(unread, int) and unread else None
            out.append(_source(f.get("id"), name,
                               f"{n} unread" if n else fallback,
                               "folder", short=name, count=n))
            if len(out) >= len(_OUTLOOK_FOLDER_NAMES) + _OUTLOOK_USER_FOLDERS:
                break
    kept = [o for o in out if o]
    if kept:
        return kept
    # A listing that could not be read — or one that answered with
    # nothing this can name — still offers something REAL to pick: the
    # three the design names are addressable by their well-known segment
    # in every locale, and `_read_mail` scopes a folder pin by exactly
    # that (`_messages_url` takes an id or a well-known name in the same
    # position). An empty picker here would read as an empty mailbox.
    return [o for o in (
        _source(sid, name, meta, "folder")
        for sid, (name, meta) in _OUTLOOK_FOLDER_NAMES.items()) if o]


async def _sources_slack(user_id: str, connector_id: str) -> list[dict]:
    content, reason = await _call(
        user_id, connector_id, "slack__list_channels",
        {"types": "public_channel,private_channel,im", "limit": 100},
    )
    if reason is not None:
        return []
    out = []
    for c in (content.get("channels") or []):
        if not isinstance(c, dict) or not c.get("is_member"):
            continue
        if c.get("type") == "im":
            who = str(c.get("user_name") or c.get("user_id") or "")
            out.append(_source(c.get("id"), who, "a direct message",
                               "channel", short=who))
            continue
        name = str(c.get("name") or "")
        members = c.get("num_members")
        # `conversations.list` carries no unread and no message count for
        # a user token, so the honest number here is who is in the room.
        meta = (f"{members} in the channel"
                if isinstance(members, int) and members
                else _clip(c.get("topic") or c.get("purpose") or "", 60))
        out.append(_source(c.get("id"), f"#{name}" if name else "",
                           meta, "channel", short=f"#{name}"))
    return [o for o in out if o]


async def _sources_teams(user_id: str, connector_id: str) -> list[dict]:
    content, reason = await _call(
        user_id, connector_id, "teams__list_chats", {"max_results": 25},
    )
    if reason is not None:
        return []
    out = []
    for c in (content.get("chats") or []):
        if not isinstance(c, dict):
            continue
        moved = _ago(_iso(c.get("last_updated_at")))
        # `thread`, not `channel`: `container_of` reads a Teams chat as a
        # thread (a chat IS the destination of `teams__send_chat_message`)
        # and a source whose kind disagrees with the reader's pin would
        # be a different object the moment it was picked.
        out.append(_source(c.get("id"), _chat_label(c),
                           f"moved {moved}" if moved else "", "thread"))
    return [o for o in out if o]


async def _sources_jira(user_id: str, connector_id: str) -> list[dict]:
    """The projects the user actually has open tickets on.

    `jira__list_projects` would list every project on the site,
    including the hundred nobody here works on. One `assignee =
    currentUser()` search names the boards that are theirs AND counts
    them in the same call — which is where "4 open · 1 due Thursday"
    comes from.
    """
    content, reason = await _call(
        user_id, connector_id, "jira__search_issues",
        {"jql": f"{_JIRA_MINE} {_JIRA_ORDER}", "max_results": 50,
         "fields": "summary,status,duedate,priority,updated,project"},
    )
    if reason is not None:
        return []
    by_project: dict[str, list] = {}
    for issue in (content.get("issues") or []):
        got = _jira_item(issue)
        if got is None or not got[0]:
            continue
        by_project.setdefault(got[0], []).append(got[1])
    return [o for o in (
        _source(proj, proj,
                _count_meta(len(items), "open", _soonest_due(items)),
                "project", count=len(items))
        for proj, items in by_project.items()) if o]


async def _sources_github(user_id: str, connector_id: str) -> list[dict]:
    content, reason = await _call(
        user_id, connector_id, "github__list_repos",
        {"sort": "pushed", "per_page": 30},
    )
    if reason is not None:
        return []
    out = []
    for r in (content.get("repos") or []):
        if not isinstance(r, dict):
            continue
        full = str(r.get("full_name") or "")
        # GitHub's `open_issues_count` counts issues AND pull requests
        # together — its own documented meaning — so the word here is
        # "open", never "PRs". Saying "2 PRs" over a number that
        # includes issues is a lie the user can check in one tap.
        n = r.get("open_issues_count")
        n = n if isinstance(n, int) else None
        pushed = _ago(_iso(r.get("pushed_at")))
        out.append(_source(
            full, full,
            _count_meta(n, "open", f"pushed {pushed}" if pushed else ""),
            "repo", short=full.split("/", 1)[-1], count=n))
    return [o for o in out if o]


async def _sources_notion(user_id: str, connector_id: str) -> list[dict]:
    content, reason = await _call(
        user_id, connector_id, "notion__search",
        {"page_size": 25, "sort": "last_edited_desc"},
    )
    if reason is not None:
        return []
    out = []
    for obj in (content.get("results") or []):
        if not isinstance(obj, dict):
            continue
        kind = _NOTION_KIND.get(str(obj.get("object") or ""))
        if kind is None:
            continue
        edited = _ago(_iso(obj.get("last_edited_time")))
        out.append(_source(obj.get("id"), obj.get("title") or "(untitled)",
                           f"edited {edited}" if edited else "", kind))
    return [o for o in out if o]


async def _sources_calendar(user_id: str, connector_id: str) -> list[dict]:
    """One source: the primary calendar.

    Not a shortcut — the connector holds `calendar.events` and nothing
    else, and `calendarList` needs `calendar.readonly`, which no user
    has ever granted (see the provider's health-probe note). Listing
    calendars the account cannot read would be a picker that writes
    nowhere, which §0.2 forbids.
    """
    from datetime import timedelta
    now = datetime.now(timezone.utc)
    content, reason = await _call(
        user_id, connector_id, "calendar__list_events",
        {"max_results": 50,
         "time_min": now.replace(microsecond=0).isoformat(),
         "time_max": (now + timedelta(days=7)).replace(
             microsecond=0).isoformat()},
    )
    if reason is not None:
        return []
    n = len([e for e in (content.get("events") or []) if isinstance(e, dict)])
    return [o for o in [_source(
        "primary", "Your calendar",
        _count_meta(n, "meetings", "this week" if n else ""),
        "folder", short="Your week", count=n)] if o]


_SOURCE_READERS = {
    "gmail": _sources_gmail,
    "outlook": _sources_outlook,
    "slack": _sources_slack,
    "teams": _sources_teams,
    "jira": _sources_jira,
    "github": _sources_github,
    "notion": _sources_notion,
    "calendar": _sources_calendar,
}


async def account_sources(
    user_id: str, connector_id: str, *, focus: Optional[list] = None,
) -> list[dict]:
    """The account's real objects, pinned ones first, at most eight.

    Pinned first for the same reason `_ordered` puts a pinned channel at
    the top of the contents sheet: the pin is where this automation
    already starts, and a picker that buries it under an alphabet reads
    as though the pin was lost.
    """
    reader = _SOURCE_READERS.get(connector_id)
    if reader is None:
        return []
    try:
        found = await asyncio.wait_for(
            reader(user_id, connector_id), _CALL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[automations] sources timed out for %s", connector_id)
        return []
    except Exception as e:  # noqa: BLE001 — a picker never raises
        logger.warning("[automations] sources failed for %s: %s: %s",
                       connector_id, type(e).__name__, str(e)[:200])
        return []
    order = {cid: n for n, (cid, _label)
             in enumerate(_pinned_containers(connector_id, focus or []))}
    found.sort(key=lambda src: order.get(src["id"], len(order)))
    return found[:_MAX_SOURCES]
