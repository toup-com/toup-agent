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
             "calendar")

_MAX_GROUPS = 6
_ITEMS_PER_GROUP = 10
_MAX_ITEMS = 60
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
) -> dict:
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
        "truncated": bool(truncated),
        "focus": list(focus or []),
    }


def _group(key: str, label: str, kind: str, *, items=None,
           reason: Optional[dict] = None, pinned: bool = False,
           pin: Optional[dict] = None) -> dict:
    return {"key": key, "label": label, "kind": kind,
            "items": list(items or []), "reason": reason,
            "pinned": bool(pinned), "pin": pin}


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

def _mail_items(messages: list, connector_id: str) -> list[dict]:
    items = []
    for m in messages[:_ITEMS_PER_GROUP]:
        if not isinstance(m, dict) or m.get("error"):
            continue
        if connector_id == "gmail":
            headers = m.get("headers") or {}
            raw_from = headers.get("From")
            subject = _clip(headers.get("Subject"), 120)
            at = _iso(headers.get("Date"))
        else:
            raw_from = m.get("from")
            subject = _clip(m.get("subject"), 120)
            at = _iso(m.get("received_at"))
        name, addr = _from_header(raw_from)
        sender = name or addr or _clip(raw_from, 80)
        snippet = _clip(m.get("snippet") or m.get("preview"), 140)
        sub = f"{sender} — {snippet}" if sender and snippet else (
            sender or snippet)
        # Gmail hands the conversation id back with every message;
        # Graph's `$select` on the list call does not, so an Outlook row
        # pins the message it is. Either way the row means ONE thread —
        # pinning a mail from Sara must not tick her other five.
        thread_id = str(m.get("threadId") or "").strip() or str(
            m.get("id") or "")
        items.append({
            "id": str(m.get("id") or ""),
            "kind": "message",
            "title": subject or "(no subject)",
            "sub": sub,
            "at": at,
            "pin": _pin("thread", thread_id, subject or sender),
        })
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
    # key, label, query, pinned, group pin
    queries: list[tuple[str, str, str, bool, Optional[dict]]] = []
    for pin in focus:
        kind, pid = pin.get("kind"), str(pin.get("id") or "").strip()
        if not pid:
            continue
        label = str(pin.get("label") or pid)
        if kind == "person":
            queries.append((f"from:{pid}", f"From {label}", f"from:{pid}",
                            True, _pin("person", pid, label)))
        elif kind in ("label", "folder"):
            queries.append((
                f"label:{pid}", label,
                f"label:{pid}" if connector_id == "gmail" else pid,
                True, _pin(kind, pid, label)))
    truncated = len(queries) > _MAX_GROUPS - 1
    queries = queries[:_MAX_GROUPS - 1]
    queries.append(("recent", "Recent", "", False, None))

    # include_body means a DIFFERENT thing per provider: outlook's False
    # still $selects subject/from/preview, gmail's False strips the list
    # to bare {id, threadId} — which is how every Gmail row rendered as
    # "(no subject)" with no sender and no time (R39). Gmail needs True
    # to get headers + snippet at all; ten parallel fetches sit well
    # inside _CALL_TIMEOUT_S.
    results = await _gather([
        _call(user_id, connector_id, tool,
              {"max_results": _ITEMS_PER_GROUP,
               "include_body": connector_id == "gmail",
               **({"query": q} if q else {})})
        for _k, _l, q, _p, _g in queries
    ])
    groups, hard_reason = [], None
    for (key, label, _q, pinned, gpin), (content, reason) in zip(
            queries, results):
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(key, label, "mailbox", reason=reason,
                                 pinned=pinned, pin=gpin))
            continue
        groups.append(_group(
            key, label, "mailbox", pinned=pinned, pin=gpin,
            items=_mail_items(content.get("messages") or [], connector_id),
        ))
    # Every group failed for the same reason ⇒ the ACCOUNT could not be
    # read, and the envelope says so rather than showing N empty rows.
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated)


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
            items.append({
                "id": ts,
                "kind": "message",
                "title": who,
                "sub": text,
                "at": _iso(ts),
                "pin": _row_pin("thread", cid, root,
                                _clip(f"{who}: {text}", 60) if text else who),
            })
        groups.append(_group(cid, label, "channel", items=items,
                             pinned=cid in is_pinned, pin=chan_pin))
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
    sub = _clip(issue.get("status") or "", 60)
    # The due date is the whole reason this list is ordered the way it
    # is, so it is stated rather than left as a bare timestamp the app
    # has to guess the meaning of.
    if due:
        sub = f"{sub} · due {due[:10]}" if sub else f"due {due[:10]}"
    return proj, {
        "id": key,
        "kind": "ticket",
        "title": f"{key} · {_clip(issue.get('summary'), 110)}" if key
        else _clip(issue.get("summary"), 110),
        "sub": sub,
        "at": due or _iso(issue.get("updated")),
        "pin": _pin("ticket", key, key),
    }


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
        if proj:
            groups.append(_group(proj, proj, "tickets", items=items,
                                 reason=group_reason.get(proj),
                                 pinned=proj in pinned,
                                 pin=_pin("project", proj, proj)))
        else:
            # A key with no dash and no project field names no project:
            # there is nothing to scope a JQL to, so no group pin.
            groups.append(_group("other", "Assigned to you", "tickets",
                                 items=items))
    # Every read failed ⇒ the ACCOUNT could not be read, and the
    # envelope says so rather than showing N empty projects.
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    if not groups:
        groups = [_group("other", "Assigned to you", "tickets", items=[])]
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated)


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
            items.append({
                "id": number,
                "kind": "pull_request",
                "title": title,
                "sub": f"#{number} · {i.get('user') or ''}".strip(" ·"),
                "at": None,
                "pin": _row_pin("ticket", key, number,
                                _clip(f"#{number} {title}", 60)),
            })
        groups.append(_group(key, key, "repo", items=items[:_ITEMS_PER_GROUP],
                             pinned=key in is_pinned, pin=repo_pin))
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=truncated)


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

    def _chat_label(c: dict) -> str:
        if c.get("topic"):
            return _clip(c["topic"], 60)
        names = [m.get("display_name") or m.get("displayName") or ""
                 for m in (c.get("members") or []) if isinstance(m, dict)]
        names = [n for n in names if n]
        return _clip(", ".join(names[:3]), 60) or "Chat"

    listed = [(str(c["id"]), _chat_label(c))
              for c in ((content or {}).get("chats") or [])
              if isinstance(c, dict) and c.get("id")]
    labels = dict(listed)
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
        for m in (content.get("messages") or [])[:_ITEMS_PER_GROUP]:
            if not isinstance(m, dict) or m.get("deleted_at"):
                continue
            body = m.get("body") or ""
            if (m.get("body_content_type") or "").lower() == "html":
                body = _strip_html(body)
            who = _clip(m.get("sender"), 80) or "(system)"
            body = _clip(body, 160)
            # A chat message has no thread of its own — Graph's chat
            # messages are flat — so the row IS the message.
            items.append({
                "id": str(m.get("id") or ""),
                "kind": "message",
                "title": who,
                "sub": body,
                "at": _iso(m.get("created_at")),
                "pin": _row_pin("thread", cid, m.get("id"),
                                _clip(f"{who}: {body}", 60) if body else who),
            })
        groups.append(_group(cid, label, "channel", items=items,
                             pinned=cid in is_pinned, pin=chat_pin))
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
        items.append({
            "id": str(ev.get("id") or ""),
            "kind": "event",
            "title": _clip(ev.get("summary"), 120) or "(untitled)",
            "sub": " · ".join(bits),
            "at": at,
            "pin": None,
        })
    return _envelope(connector_id, focus=focus, groups=[
        _group("upcoming", "Coming up", "events", items=items),
    ])


_READERS = {
    "gmail": _read_mail,
    "outlook": _read_mail,
    "slack": _read_slack,
    "jira": _read_jira,
    "github": _read_github,
    "teams": _read_teams,
    "calendar": _read_calendar,
}


# ── the entry point ──────────────────────────────────────────────────

async def account_contents(
    user_id: str, *, connector_id: str, focus: Optional[list] = None,
    connection: Optional[dict] = None,
) -> dict:
    """What is inside one account, in the uniform envelope.

    `connection` is this user's connection-state row for the connector
    (the caller already has it); when it says the account is not usable
    we answer with THAT reason rather than making a call we know will
    fail — a "reconnect" sentence beats a provider's 401 relayed
    through three layers.
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
            return _envelope(connector_id, focus=focus, reason={
                "code": "not_connected",
                "sentence": f"{name} is not connected.",
                "retryable": False,
                "consent_url": f"/api/oauth/connect/{connector_id}",
            })
        if status and status != "active":
            return _envelope(connector_id, focus=focus, reason={
                "code": "reconnect",
                "sentence": f"{name} needs signing in again before it "
                            f"can be read.",
                "retryable": False,
                "consent_url": f"/api/oauth/connect/{connector_id}",
            })
    reader = _READERS.get(connector_id)
    if reader is None:
        return _envelope(connector_id, focus=focus, reason={
            "code": "not_supported",
            "sentence": f"There is no way to look inside {name} yet.",
            "retryable": False,
        })
    try:
        env = await asyncio.wait_for(
            reader(user_id, connector_id, focus), _TOTAL_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[automations] contents timed out for %s", connector_id)
        return _envelope(connector_id, focus=focus, reason={
            "code": "unreachable",
            "sentence": f"{name} took too long to answer. Try again.",
            "retryable": True,
        })
    except Exception as e:  # noqa: BLE001 — the reason IS the answer
        logger.warning("[automations] contents failed for %s: %s: %s",
                       connector_id, type(e).__name__, str(e)[:200])
        return _envelope(connector_id, focus=focus, reason={
            "code": "unreachable",
            "sentence": f"Could not look inside {name} just now.",
            "retryable": True,
        })
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
    return env
