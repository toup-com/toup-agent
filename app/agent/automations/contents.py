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
from datetime import datetime, timezone
from typing import Any, Optional

from app.services import automation_verbs as verbs

from . import registry as reg

logger = logging.getLogger(__name__)

#: Connectors with a reader. Everything else answers `not_supported`
#: BY NAME — an unnamed refusal reads as a bug.
SUPPORTED = ("gmail", "outlook", "slack", "jira", "github")

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
           reason: Optional[dict] = None, pinned: bool = False) -> dict:
    return {"key": key, "label": label, "kind": kind,
            "items": list(items or []), "reason": reason,
            "pinned": bool(pinned)}


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
            sender = _clip(headers.get("From"), 80)
            subject = _clip(headers.get("Subject"), 120)
            at = _iso(headers.get("Date"))
        else:
            sender = _clip(m.get("from"), 80)
            subject = _clip(m.get("subject"), 120)
            at = _iso(m.get("received_at"))
        snippet = _clip(m.get("snippet") or m.get("preview"), 140)
        sub = f"{sender} — {snippet}" if sender and snippet else (
            sender or snippet)
        items.append({
            "id": str(m.get("id") or ""),
            "kind": "message",
            "title": subject or "(no subject)",
            "sub": sub,
            "at": at,
        })
    return items


async def _read_mail(user_id: str, connector_id: str, focus: list) -> dict:
    """gmail / outlook — recent messages: sender, subject, snippet.

    A pinned person or label becomes its OWN group with its own query,
    because that is the question the pin was made to ask ("what is in
    here from Sara?"), not a filter applied to a list of everything.
    """
    tool = f"{connector_id}__list_messages"
    queries: list[tuple[str, str, str, bool]] = []   # key,label,query,pinned
    for pin in focus[:_MAX_GROUPS]:
        kind, pid = pin.get("kind"), str(pin.get("id") or "")
        label = pin.get("label") or pid
        if kind == "person":
            queries.append((f"from:{pid}", f"From {label}",
                            f"from:{pid}", True))
        elif kind in ("label", "folder"):
            queries.append((f"label:{pid}", label,
                            f"label:{pid}" if connector_id == "gmail" else pid,
                            True))
    if not queries:
        queries = [("recent", "Recent", "", False)]

    results = await _gather([
        _call(user_id, connector_id, tool,
              {"max_results": _ITEMS_PER_GROUP, "include_body": False,
               **({"query": q} if q else {})})
        for _k, _l, q, _p in queries
    ])
    groups, hard_reason = [], None
    for (key, label, _q, pinned), (content, reason) in zip(queries, results):
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(key, label, "mailbox", reason=reason,
                                 pinned=pinned))
            continue
        groups.append(_group(
            key, label, "mailbox", pinned=pinned,
            items=_mail_items(
                [m for m in (content.get("messages") or [])], connector_id),
        ))
    # Every group failed for the same reason ⇒ the ACCOUNT could not be
    # read, and the envelope says so rather than showing N empty rows.
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups)


async def _read_slack(user_id: str, connector_id: str, focus: list) -> dict:
    """slack — recent messages per JOINED conversation.

    `is_member` is the filter, exactly as `automations__list_targets`
    filters it (R38): the manifest has no `chat:write.public`, so a
    channel the workspace never joined is neither readable nor
    postable, and offering it is how #general got named.
    """
    pinned = [p for p in focus if p.get("kind") in ("channel", "thread")]
    channels: list[tuple[str, str, bool]] = [
        (str(p.get("id")), str(p.get("label") or p.get("id")), True)
        for p in pinned[:_MAX_GROUPS]
    ]
    if not channels:
        content, reason = await _call(
            user_id, connector_id, "slack__list_channels",
            {"types": "public_channel,private_channel", "limit": 100},
        )
        if reason is not None:
            return _envelope(connector_id, focus=focus, reason=reason)
        rows = [c for c in (content.get("channels") or [])
                if isinstance(c, dict) and c.get("is_member") and c.get("id")]
        channels = [
            (str(c["id"]), f"#{c.get('name')}" if c.get("name")
             else str(c.get("user_name") or c["id"]), False)
            for c in rows[:_MAX_GROUPS]
        ]
        if not channels:
            # Joined nothing is a real, readable answer — one empty
            # group, not a reason. The user has a Slack; they are in no
            # channel this automation could start from.
            return _envelope(connector_id, focus=focus, groups=[])

    results = await _gather([
        _call(user_id, connector_id, "slack__read_messages",
              {"channel": cid, "limit": _ITEMS_PER_GROUP})
        for cid, _label, _p in channels
    ])
    groups, hard_reason = [], None
    for (cid, label, is_pin), (content, reason) in zip(channels, results):
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(cid, label, "channel", reason=reason,
                                 pinned=is_pin))
            continue
        items = []
        for m in (content.get("messages") or [])[:_ITEMS_PER_GROUP]:
            if not isinstance(m, dict):
                continue
            items.append({
                "id": str(m.get("ts") or ""),
                "kind": "message",
                "title": _clip(m.get("from"), 80) or "(app)",
                "sub": _clip(m.get("text"), 160),
                "at": _iso(m.get("ts")),
            })
        groups.append(_group(cid, label, "channel", items=items,
                             pinned=is_pin))
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups)


async def _read_jira(user_id: str, connector_id: str, focus: list) -> dict:
    """jira — the user's own open tickets, soonest due first."""
    projects = [str(p.get("id")) for p in focus
                if p.get("kind") == "project" and p.get("id")]
    scope = ""
    if projects:
        keys = ", ".join(f'"{p}"' for p in projects[:_MAX_GROUPS])
        scope = f"project in ({keys}) AND "
    jql = (f"{scope}assignee = currentUser() AND resolution = Unresolved "
           f"ORDER BY duedate ASC, updated DESC")
    content, reason = await _call(
        user_id, connector_id, "jira__search_issues",
        {"jql": jql, "max_results": _ITEMS_PER_GROUP * 2,
         "fields": "summary,status,duedate,priority,updated,project"},
    )
    if reason is not None:
        return _envelope(connector_id, focus=focus, reason=reason)
    items, overdue = [], []
    for i in (content.get("issues") or []):
        if not isinstance(i, dict):
            continue
        due = _iso(i.get("duedate"))
        row = {
            "id": str(i.get("key") or ""),
            "kind": "ticket",
            "title": f"{i.get('key')} · {_clip(i.get('summary'), 110)}",
            "sub": _clip(i.get("status") or "", 60),
            "at": due or _iso(i.get("updated")),
        }
        # The due date is the whole reason this list is ordered the way
        # it is, so it is stated rather than left as a bare timestamp
        # the app has to guess the meaning of.
        if due:
            row["sub"] = (f"{row['sub']} · due {due[:10]}" if row["sub"]
                          else f"due {due[:10]}")
            overdue.append(row)
        else:
            items.append(row)
    groups = []
    if overdue:
        groups.append(_group("dated", "With a due date", "tickets",
                             items=overdue[:_ITEMS_PER_GROUP],
                             pinned=bool(projects)))
    if items:
        groups.append(_group("undated", "No due date", "tickets",
                             items=items[:_ITEMS_PER_GROUP],
                             pinned=bool(projects)))
    if not groups:
        groups = [_group("dated", "Assigned to you", "tickets", items=[])]
    return _envelope(connector_id, focus=focus, groups=groups,
                     truncated=len(overdue) > _ITEMS_PER_GROUP
                     or len(items) > _ITEMS_PER_GROUP)


async def _read_github(user_id: str, connector_id: str, focus: list) -> dict:
    """github — open pull requests, per repository.

    `github__list_issues` is the only listing tool the manifest has and
    it returns issues AND pull requests, flagged; the PRs are filtered
    out HERE rather than asked for, because there is no tool that asks.
    """
    repos = [str(p.get("id")) for p in focus
             if p.get("kind") == "repo" and p.get("id")]
    if not repos:
        content, reason = await _call(
            user_id, connector_id, "github__list_repos",
            {"sort": "pushed", "per_page": 30},
        )
        if reason is not None:
            return _envelope(connector_id, focus=focus, reason=reason)
        repos = [str(r.get("full_name")) for r in (content.get("repos") or [])
                 if isinstance(r, dict) and r.get("full_name")][:3]
        if not repos:
            return _envelope(connector_id, focus=focus, groups=[])
    pinned = bool([p for p in focus if p.get("kind") == "repo"])

    pairs = [tuple(r.split("/", 1)) for r in repos[:_MAX_GROUPS]
             if "/" in r]
    results = await _gather([
        _call(user_id, connector_id, "github__list_issues",
              {"owner": owner, "repo": repo, "state": "open", "per_page": 30})
        for owner, repo in pairs
    ])
    groups, hard_reason = [], None
    for (owner, repo), (content, reason) in zip(pairs, results):
        key = f"{owner}/{repo}"
        if reason is not None:
            hard_reason = hard_reason or reason
            groups.append(_group(key, key, "repo", reason=reason,
                                 pinned=pinned))
            continue
        items = [
            {
                "id": str(i.get("number") or ""),
                "kind": "pull_request",
                "title": _clip(i.get("title"), 120),
                "sub": f"#{i.get('number')} · {i.get('user') or ''}".strip(
                    " ·"),
                "at": None,
            }
            for i in (content.get("issues") or [])
            if isinstance(i, dict) and i.get("is_pull_request")
        ]
        groups.append(_group(key, key, "repo",
                             items=items[:_ITEMS_PER_GROUP], pinned=pinned))
    if hard_reason is not None and all(g["reason"] for g in groups):
        return _envelope(connector_id, focus=focus, reason=hard_reason)
    return _envelope(connector_id, focus=focus, groups=groups)


_READERS = {
    "gmail": _read_mail,
    "outlook": _read_mail,
    "slack": _read_slack,
    "jira": _read_jira,
    "github": _read_github,
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
    reader = _READERS.get(connector_id)
    if reader is None:
        return _envelope(connector_id, focus=focus, reason={
            "code": "not_supported",
            "sentence": f"There is no way to look inside {name} yet.",
            "retryable": False,
        })
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
