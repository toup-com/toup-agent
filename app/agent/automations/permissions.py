"""Per-automation permissions — the one source the canvas reads (R30 §4.4).

Permission ids are stable strings `"{connector_id}.{suffix}"` derived
from C's verb-entry tables (`app/services/automation_verb_entries.py`,
`permission_labels` + `rails`) with total fallbacks, mapped server-side
to capability metadata + the per-automation grant. Three kinds:

  - read  (suffix starts with `read`)   — what the automation may look at
  - write (any other labeled suffix)    — grant-gated; starts UNGRANTED
  - rail  (from the connector's `rails`)— can NEVER be allowed
    (`409 hard_rail`); rendered in IT CANNOT without the tap affordance

Resolution order (the canvas's `permOf`): the saved
`automation_account_permissions` row, else the connector default —
reads in `can`; writes in `can` only when an approved grant backs them;
rails always in `cant`. Every reader (workflow, account sheet,
connector sheet with `automation_id`, captions, badges) resolves
through here, so one save repaints them all.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Automation, AutomationAccountPermission

logger = logging.getLogger(__name__)


class PermissionError409(Exception):
    """A refused permission write — `code` is the wire 409 code."""

    def __init__(self, code: str, sentence: str, extra: Optional[dict] = None):
        super().__init__(sentence)
        self.code = code
        self.sentence = sentence
        self.extra = extra or {}


# ------------------------------------------------- the per-connector catalog

_FALLBACK_LABELS: dict[str, dict[str, str]] = {
    # Used only when C's entries module is absent — totality, not polish.
    "gmail": {"read_new_mail": "Read new mail", "write_drafts": "Write drafts"},
    "outlook": {"read_new_mail": "Read new mail", "write_drafts": "Write drafts"},
    "slack": {"read_channels": "Read your channels", "write_post": "Post as you"},
    "jira": {"read_board": "Read your tickets", "write_comment": "Comment"},
    "github": {"read_prs": "Read pull requests"},
    "teams": {"read_chats": "Read your chats", "write_post": "Post as you"},
    "notion": {"read_pages": "Read pages", "write_notes": "Write notes"},
    "drive": {"read_files": "Read your files"},
    "docs": {"read_docs": "Read documents", "write_append": "Add to a doc"},
    "calendar": {"read_week": "Read your week"},
    "stub": {"read_feed": "Read the test feed", "write_post": "Post to the test channel"},
}
_FALLBACK_RAILS: dict[str, tuple] = {
    "gmail": ("Send anything", "Delete mail"),
    "outlook": ("Send anything", "Delete mail"),
    "slack": ("Read private DMs",),
    "github": ("Push or merge",),
    "calendar": ("Invite other people",),
    "jira": ("Close or reassign",),
    "notion": ("Delete pages",),
    "teams": ("Send outside your chats",),
    "drive": ("Delete files",),
    "docs": ("Delete documents",),
    "stub": ("Send anything",),
}


def _entries():
    try:
        from app.services import automation_verb_entries as entries
        return entries
    except Exception:  # noqa: BLE001 — C's module may not be merged yet
        return None


def _rail_id(connector_id: str, label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")[:32]
    return f"{connector_id}.rail_{slug}"


def catalog_for(connector_id: str) -> dict:
    """`{"reads": [{id,label}], "writes": [{id,label}], "rails": [{id,label}]}`
    for one connector. Total — an unknown connector gets one generic
    read permission and no writes."""
    e = _entries()
    labels: dict[str, str] = {}
    rails: tuple = ()
    if e is not None:
        labels = dict((getattr(e, "V2_PERMISSION_LABELS", {}) or {})
                      .get(connector_id) or {})
        rails = tuple((getattr(e, "V2_RAILS", {}) or {})
                      .get(connector_id) or ())
    if not labels:
        labels = dict(_FALLBACK_LABELS.get(connector_id) or {})
    if not rails:
        rails = _FALLBACK_RAILS.get(connector_id, ())
    if not labels:
        labels = {"read": "Read"}
    reads, writes = [], []
    for suffix, label in labels.items():
        entry = {"id": f"{connector_id}.{suffix}", "label": label}
        (reads if suffix.startswith("read") else writes).append(entry)
    return {
        "reads": reads,
        "writes": writes,
        "rails": [{"id": _rail_id(connector_id, r), "label": r}
                  for r in rails],
    }


def is_rail_id(permission_id: str) -> bool:
    return ".rail_" in permission_id


def is_read_id(permission_id: str) -> bool:
    _, _, suffix = permission_id.partition(".")
    return suffix.startswith("read")


# -------------------------------------------------------------- resolution

async def saved_row(
    db: AsyncSession, automation_id: str, account_id: str,
) -> Optional[AutomationAccountPermission]:
    return (
        await db.execute(
            select(AutomationAccountPermission).where(
                AutomationAccountPermission.automation_id == automation_id,
                AutomationAccountPermission.account_id == account_id,
            )
        )
    ).scalar_one_or_none()


def _default_can_write(automation: Automation, connector_id: str) -> bool:
    """A write defaults to CAN only when an approved grant backs a write
    step on this connector (grant_id snapshotted into the spec)."""
    try:
        raw = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        return False
    if raw.get("version") != 2:
        action = raw.get("action") or {}
        return bool(action.get("tool")) and \
            action.get("connector_id") == connector_id
    for s in raw.get("steps") or []:
        if s.get("connector_id") == connector_id and s.get("grant_id"):
            return True
    return False


async def resolve(
    db: AsyncSession, *, automation: Automation, account_id: str,
) -> dict:
    """The §4.4 `can`/`cant` lists for one (automation, account) — the
    saved set else the connector default; rails always in `cant`."""
    cat = catalog_for(account_id)
    row = await saved_row(db, automation.id, account_id)
    if row is not None:
        try:
            can_ids = set(json.loads(row.can_json or "[]"))
            cant_ids = set(json.loads(row.cant_json or "[]"))
        except (ValueError, TypeError):
            can_ids, cant_ids = set(), set()
    else:
        can_ids = {p["id"] for p in cat["reads"]}
        if _default_can_write(automation, account_id):
            can_ids |= {p["id"] for p in cat["writes"]}
        cant_ids = set()
    can, cant = [], []
    for p in cat["reads"] + cat["writes"]:
        if p["id"] in can_ids and p["id"] not in cant_ids:
            can.append({"id": p["id"], "label": p["label"]})
        else:
            cant.append({"id": p["id"], "label": p["label"],
                         "kind": "ungranted"})
    for p in cat["rails"]:
        cant.append({"id": p["id"], "label": p["label"], "kind": "rail"})
    return {"can": can, "cant": cant}


def write_grant_ids(automation: Automation, connector_id: str) -> list[str]:
    """The grant ids the spec's write steps on this connector carry."""
    try:
        raw = json.loads(automation.spec_json or "{}")
    except (ValueError, TypeError):
        return []
    if raw.get("version") != 2:
        action = raw.get("action") or {}
        gid = action.get("grant_id")
        return [gid] if gid and action.get(
            "connector_id") == connector_id else []
    return [s["grant_id"] for s in raw.get("steps") or []
            if s.get("connector_id") == connector_id and s.get("grant_id")]


async def has_approved_write_grant(
    *, automation: Automation, user_id: str, connector_id: str,
) -> bool:
    """Does an APPROVED platform grant back a write on this connector?

    AUDIT-2: the caller used to read the connector's OAuth `scopes` off
    the connection state. Scopes say the user once let the platform hold
    a token with that reach; a grant says the user approved THIS
    automation to make THAT call to THAT target. Every connected Slack
    carries a write scope, so the green ✓ accepted "Post as you" for an
    automation nobody had granted anything — the §3.7 consent flow the
    409 exists to trigger never ran, and the sheet showed the permission
    as allowed until the dispatcher failed closed at fire time.

    Fails CLOSED: an unreachable platform means no approved grant, which
    returns the user to consent rather than through it.
    """
    ids = write_grant_ids(automation, connector_id)
    if not ids:
        return False
    from . import registry as _reg
    for gid in ids:
        grant = await _reg.fetch_grant(user_id, gid)
        # Same predicate the compiler arms on (compiler.py:347).
        if (grant or {}).get("status") == "approved":
            return True
    return False


async def revoke_writes(
    db: AsyncSession, *, automation: Automation, connector_id: str,
) -> list[str]:
    """Move every allowed write for one connector into IT CANNOT.

    AUDIT-4: revoking a grant paused the automation but left the saved
    permission row untouched, so the account sheet kept the write in IT
    CAN — the one surface whose whole job is to answer "what may this
    thing do?" answered with a permission the platform had already taken
    away. Writing an EXPLICIT row matters: the spec keeps its `grant_id`
    after a revoke, so the unsaved default would still resolve to CAN.
    """
    cat = catalog_for(connector_id)
    write_ids = {p["id"] for p in cat["writes"]}
    if not write_ids:
        return []
    current = await resolve(db, automation=automation,
                            account_id=connector_id)
    can_ids = [p["id"] for p in current["can"]]
    moved = sorted(pid for pid in can_ids if pid in write_ids)
    if not moved:
        return []
    row = await saved_row(db, automation.id, connector_id)
    if row is None:
        row = AutomationAccountPermission(
            automation_id=automation.id, account_id=connector_id,
        )
        db.add(row)
    try:
        prev_cant = set(json.loads(row.cant_json or "[]"))
    except (ValueError, TypeError):
        prev_cant = set()
    row.can_json = json.dumps(
        sorted(pid for pid in can_ids if pid not in write_ids))
    row.cant_json = json.dumps(sorted(prev_cant | set(moved)))
    row.updated_at = datetime.utcnow()
    await db.commit()
    logger.info("[automations] grant revoke demoted %d write(s) on %s "
                "for %s", len(moved), connector_id, automation.id)
    return moved


async def save(
    db: AsyncSession, *, automation: Automation, account_id: str,
    can_ids: list[str], cant_ids: list[str], has_write_grant: bool,
) -> dict:
    """The green-✓ commit (§4.4). Raises PermissionError409:
    `hard_rail` for a rail in `can`; `last_read` when the save would
    remove the account's last read; `needs_consent` when a write id is
    allowed without an approved grant behind it."""
    cat = catalog_for(account_id)
    known = {p["id"] for p in cat["reads"] + cat["writes"]}
    can_set = [p for p in can_ids if p in known or is_rail_id(p)]
    for pid in can_set:
        if is_rail_id(pid):
            raise PermissionError409(
                "hard_rail", "It can never do this.")
    read_ids = {p["id"] for p in cat["reads"]}
    if read_ids and not (read_ids & set(can_set)):
        raise PermissionError409(
            "last_read", "Take away the account instead")
    write_allowed = [p for p in can_set
                     if p in {w["id"] for w in cat["writes"]}]
    if write_allowed and not has_write_grant:
        # §4.4 shape: `consent:{connector_id, mode, scopes}` — the app
        # reads the NESTED object to run the §3.7 flow and retry. It
        # was served flat, so the app had nothing to run and the retry
        # loop had no starting point. Flat keys stay alongside it: a
        # shipped build reading them keeps working.
        try:
            spec = json.loads(automation.spec_json or "{}")
        except (ValueError, TypeError):
            spec = {}
        labels = {w["id"]: w["label"] for w in cat["writes"]}
        raise PermissionError409(
            "needs_consent",
            "Allowing this needs your say-so first.",
            {"consent": {
                "connector_id": account_id,
                "mode": (spec.get("mode") or "auto"),
                "scopes": [{"id": p, "label": labels.get(p, p)}
                           for p in write_allowed],
             },
             "connector_id": account_id, "permissions": write_allowed},
        )
    row = await saved_row(db, automation.id, account_id)
    if row is None:
        row = AutomationAccountPermission(
            automation_id=automation.id, account_id=account_id,
        )
        db.add(row)
    row.can_json = json.dumps(sorted(set(can_set) - set(cant_ids)))
    row.cant_json = json.dumps(
        sorted({p for p in cant_ids if p in known}))
    row.updated_at = datetime.utcnow()
    await db.commit()
    return await resolve(db, automation=automation, account_id=account_id)
