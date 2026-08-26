"""One account state, one reason, one fix — CONTRACTS-R31 §4.4.

Before this module there were FOUR vocabularies for "is this connector
working", and they narrowed at every hop until nothing actionable was
left:

  `ConnectorIdentity.status`     active | reauth_required | revoked | provider_down
  `connector.state` frame        connected | expired
  the workflow account sheet     connected | expired | missing → expired
  the run's tool turn            prose, from a substring match, thrown away

So on 26 August the Connectors page said `Connected · 10` while the same
Outlook account's own sheet said `Last use · Could not connect · access
expired`, and the agent — asked in the thread — knew the real reasons
(the GitHub org had not approved Toup, Outlook had no mail-read scope,
Slack had no `channels:history`) that the run ledger had never been told.

This module is the single derivation. It answers, for one account:

    {account_state, reason_code, fix, checked_at, source}

and every surface reads it: `/summary`, `/thread`, `/workflow`, the
account card, `/connectors`, the picker, the home card meta.

Two rules that are code, not copy, and are load-bearing:

1. **A transient failure keeps `connected`.** `rate_limited`,
   `vendor_down` and `timeout` say nothing about the credential — they
   say the network had a bad minute. Moving an account to `expired` for
   a timeout teaches the user to ignore the word. Only
   `token_expired`, `token_revoked`, `scope_missing:*` and
   `org_approval_needed` may move it, and that is what
   `test_health_is_the_ledger` asserts in both directions.

2. **The last REAL USE outranks everything.** A token that refreshes
   cleanly and then fails every call is `active` by the vault's reading
   and useless by the user's. The precedence is: last real use → the
   OAuth/vault flip → a scope probe → the identity row.

The strings all live in `fixtures/automations/reason-strings.json`
(C's). This module maps a reason to a row of that table and never
writes a user-facing sentence of its own.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

# §4.4 vocabularies. Imported by the ledger's `needs_you` validator too.
ACCOUNT_STATES = (
    "connected", "expired", "revoked", "scope_missing",
    "org_approval_needed", "not_connected",
)
FIXES = ("reconnect", "grant", "approve", "connect", "retry")

# Reasons that describe the MOMENT, not the credential. An account whose
# last failure was one of these stays `connected` and offers `Try again`.
TRANSIENT_REASONS = frozenset({"rate_limited", "vendor_down", "timeout"})

# reason_code → (account_state, fix). `scope_missing` carries its scope
# after a colon (`scope_missing:Mail.Read`) and is matched on the head.
_REASON_MAP: dict[str, tuple[str, str]] = {
    "token_expired": ("expired", "reconnect"),
    "token_revoked": ("revoked", "reconnect"),
    "scope_missing": ("scope_missing", "grant"),
    "org_approval_needed": ("org_approval_needed", "approve"),
    "not_connected": ("not_connected", "connect"),
    "rate_limited": ("connected", "retry"),
    "vendor_down": ("connected", "retry"),
    "timeout": ("connected", "retry"),
}

# The engine's older, looser failure tokens (`executor_v2._failure_reason`
# and the RPC envelope kinds) mapped onto R31 reason codes. `unreachable`
# is the default that token function returns and it is genuinely
# ambiguous — it means "the call did not come back", which is transient
# until something says otherwise.
_LEGACY_REASON_ALIASES: dict[str, str] = {
    "reauth_required": "token_expired",
    "access_expired": "token_expired",
    "revoked": "token_revoked",
    "provider_down": "vendor_down",
    "unreachable": "timeout",
    "": "timeout",
}

# Vendor-side signals that mean "your org has not let this app in". Kept
# narrow: an ordinary 403 is a scope problem, and telling a user to go
# ask their GitHub owner when the real fix is one tap is worse than
# saying nothing.
_ORG_APPROVAL_RE = re.compile(
    r"(oauth app|organization|organisation).{0,40}"
    r"(approv|polic|restrict|not authorized|denied access)",
    re.IGNORECASE | re.DOTALL,
)
_SCOPE_RE = re.compile(
    r"(insufficient|missing|required)[ _-]?(scope|permission)s?"
    r"|scope_missing|needs? more access",
    re.IGNORECASE,
)

PROBE_CACHE_S = 600  # ten minutes — vendor rate limits, §4.4


# ── the string table ─────────────────────────────────────────────────

_TABLE: Optional[dict] = None


def _table_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    return os.path.join(root, "fixtures", "automations",
                        "reason-strings.json")


def strings() -> dict:
    """C's `account_state × surface` table. Loaded once; a missing or
    unreadable file yields `{}` rather than raising — a run must never
    die because a copy fixture moved."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    try:
        with open(_table_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _TABLE = data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.warning("[account_health] reason-strings unreadable: %s", e)
        _TABLE = {}
    return _TABLE


def form(key: str, **values: Any) -> str:
    """One `forms` entry, rendered. Unknown key ⇒ empty string."""
    tmpl = (strings().get("forms") or {}).get(key) or ""
    return render(tmpl, values)


def render(template: str, values: dict) -> str:
    """Substitute `{slot}`s, and DROP any clause whose slot has no value.

    R31-25: `{need_count}` reached a user's job sheet because the only
    interpolator on that path substituted `{n}` and `{count}` and passed
    everything else through verbatim. A renderer that cannot fill a slot
    must remove the clause, never show the brace — and the caller must
    not have to know which slots a string happens to contain.
    """
    if not template:
        return ""
    out = template
    for key, value in (values or {}).items():
        if value is None or value == "":
            continue
        out = out.replace("{" + str(key) + "}", str(value))
    if "{" not in out:
        return out.strip()
    # Something is still unfilled. Drop the clause it sits in — a
    # sentence is delimited by ` · `, then by `, `, then by nothing.
    for sep in (" · ", ", "):
        if sep in out:
            kept = [c for c in out.split(sep) if "{" not in c]
            out = sep.join(kept)
            if "{" not in out:
                return out.strip()
    # A whole-sentence template with an unfillable slot: say nothing
    # rather than say a brace.
    logger.warning("automation.copy.unfilled template=%r", template[:120])
    return ""


def connector_block(connector_id: str) -> dict:
    return ((strings().get("per_connector") or {})
            .get(connector_id) or {})


def display_of(connector_id: str, fallback: str = "") -> str:
    """The connector's own capitalisation, from C's table."""
    return connector_block(connector_id).get("display") or fallback \
        or connector_id


def slots_for(connector_id: str, name: str = "", **extra: Any) -> dict:
    """Everything C's templates may name, for one connector.

    `{Connector}` is the display name and `{capability}` is that
    connector's own phrase for what it does ("read your repositories").
    A slot the caller cannot fill is dropped with its clause by
    `render` — never shown, and never guessed.
    """
    block = connector_block(connector_id)
    out: dict[str, Any] = {
        "Connector": block.get("display") or name or connector_id,
        "capability": block.get("capability") or "",
    }
    out.update({k: v for k, v in (extra or {}).items() if v not in (None, "")})
    return out


def sentence_for(
    *, account_state: str, reason_code: str, connector_id: str,
    name: str = "", surface: str = "thread_sentence", **extra: Any,
) -> str:
    """The one string for this (state, reason, surface).

    Precedence is C's: a per-connector override for this state wins
    over the generic state row — GitHub's org-approval wording is
    written once, in GitHub's block, and every surface that names it
    reads the same sentence.
    """
    table = strings()
    # THREE levels, in this order (C's shape):
    #
    #   1. `reason_codes[code]` — the only place that knows WHICH
    #      transient thing happened. A rate limit, an outage and a
    #      timeout all leave the account `connected`, so `states` has
    #      nothing to say about them and correctly says nothing; the
    #      sentence lives with the reason. Reading `states` first is why
    #      a timed-out account rendered a bare "I could not read GitHub."
    #   2. `per_connector[cid][state]` — GitHub's org-approval wording.
    #   3. `states[state]` — the generic row.
    reason_row = reason_block(reason_code)
    template = reason_row.get(surface)
    if template is None:
        template = (connector_block(connector_id).get(account_state)
                    or {}).get(surface)
    if template is None:
        template = ((table.get("states") or {}).get(account_state)
                    or {}).get(surface)
    return render(template or "", slots_for(connector_id, name, **extra))


def fix_button(fix: str, connector_id: str, name: str = "") -> str:
    """The button's label for one fix kind — one entry point for all
    four callers (the thread card, the E-1 line, the home action, the
    Connectors page), so they can never disagree."""
    tmpl = ((strings().get("fix_buttons") or {}).get(fix)) or ""
    return render(tmpl, slots_for(connector_id, name))


def join_names(names: list[str]) -> str:
    """`{A}` / `{A} and {B}` / `{LIST} and {LAST}` — C's `names_join`.

    EVERY name, at any count (R31-07). The old home-card form took
    `names[0]` and dropped the rest, so a user fixed one account,
    re-ran, and met the next one.
    """
    clean = [n for n in (names or []) if n]
    if not clean:
        return ""
    join = strings().get("names_join") or {}
    if len(clean) == 1:
        return render(join.get("one") or "{A}", {"A": clean[0]})
    if len(clean) == 2:
        return render(join.get("two") or "{A} and {B}",
                      {"A": clean[0], "B": clean[1]})
    return render(join.get("many") or "{LIST} and {LAST}",
                  {"LIST": ", ".join(clean[:-1]), "LAST": clean[-1]})


def names_sentence(names: list[str], *, prefix: str) -> str:
    """One of C's `{prefix}_1 / _2 / _n` forms, filled."""
    clean = [n for n in (names or []) if n]
    if not clean:
        return ""
    if len(clean) == 1:
        return form(f"{prefix}_1", A=clean[0])
    if len(clean) == 2:
        return form(f"{prefix}_2", A=clean[0], B=clean[1])
    return form(f"{prefix}_n", LIST=", ".join(clean[:-1]), LAST=clean[-1])


# ── the derivation ───────────────────────────────────────────────────

def classify(reason: Optional[str], message: str = "") -> str:
    """Turn whatever the failure gave us into an R31 reason code.

    `reason` is the engine's token (`_failure_reason`) or an RPC
    envelope kind; `message` is the provider text, which is the only
    place an org-approval refusal announces itself.
    """
    token = (reason or "").strip()
    if token.startswith("scope_missing"):
        return token                      # already carries its scope
    if _ORG_APPROVAL_RE.search(message or ""):
        return "org_approval_needed"
    if _SCOPE_RE.search(message or ""):
        return "scope_missing"
    return _LEGACY_REASON_ALIASES.get(token, token) or "timeout"


def reason_block(reason_code: str) -> dict:
    head = (reason_code or "").split(":", 1)[0]
    row = (strings().get("reason_codes") or {}).get(head)
    return row if isinstance(row, dict) else {}


def state_for_reason(reason_code: str) -> tuple[str, str]:
    """`(account_state, fix)` for one reason code. Total.

    C's table is authoritative where it has the row; the built-in map
    is the fallback so this stays total if the fixture moves. The
    TRANSIENT rule is code either way (§4.4): a reason that says
    nothing about the credential may not move the account off
    `connected`, whatever any table says.
    """
    head = (reason_code or "").split(":", 1)[0]
    row = reason_block(head)
    if row.get("state") and row.get("fix"):
        return str(row["state"]), str(row["fix"])
    return _REASON_MAP.get(head, ("connected", "retry"))


def is_transient(reason_code: str) -> bool:
    head = (reason_code or "").split(":", 1)[0]
    row = reason_block(head)
    if "transient" in row:
        return bool(row["transient"])
    return head in TRANSIENT_REASONS


def sheet_detail(reason_code: str) -> str:
    """The RUN ROW's detail for a failed account ("access expired").

    Deliberately slot-free: the run row is rendered by the verb
    dictionary, which knows the connector and nothing else. The
    connector sheet's richer subtitle lives in `states[*]` and may name
    the capability, because that surface knows which connector it is
    drawing.
    """
    return str(reason_block(reason_code).get("sheet_detail") or "")


def needs_you_payload(
    *, account_id: str, connector_id: str, name: str,
    reason_code: str, approval_url: Optional[str] = None,
) -> dict:
    """The `needs_you` turn body (§4.4/§4.5) — the card the thread shows
    for one failed source, with the button that actually fixes it."""
    account_state, fix = state_for_reason(reason_code)
    display = display_of(connector_id, name)
    sentence = sentence_for(
        account_state=account_state, reason_code=reason_code,
        connector_id=connector_id, name=display,
    )
    label = sentence_for(
        account_state=account_state, reason_code=reason_code,
        connector_id=connector_id, name=display, surface="button_label",
    ) or fix_button(fix, connector_id, display) or "Try again"
    return {
        "account_id": account_id,
        "connector_id": connector_id,
        "name": display,
        "reason_code": reason_code,
        "sentence": sentence or f"I could not read {display}.",
        "fix": fix,
        "fix_label": label,
        "approval_url": approval_url,
    }


async def record_use(
    db, *, user_id: str, account_id: str, ok: bool,
    reason_code: str = "", message: str = "",
) -> None:
    """Write the LAST REAL USE for this account.

    This is the input the old health story never had: a tool call that
    actually happened, and what it said. A successful call clears the
    account; a failing one records the reason so every surface can say
    the same thing about it within the same run.

    Best-effort by contract — health is a projection, and a projection
    that fails must not take the run with it.
    """
    from app.db.models import AccountHealth

    code = classify(reason_code, message) if not ok else ""
    account_state, fix = (
        ("connected", "retry") if ok else state_for_reason(code)
    )
    now = datetime.utcnow()
    try:
        from sqlalchemy import select
        row = (await db.execute(
            select(AccountHealth).where(
                AccountHealth.user_id == user_id,
                AccountHealth.account_id == account_id,
            )
        )).scalar_one_or_none()
        if row is None:
            row = AccountHealth(user_id=user_id, account_id=account_id)
            db.add(row)
        row.state = account_state
        row.reason_code = code
        row.fix = fix
        row.checked_at = now
        row.source = "use"
        await db.flush()
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.debug("[account_health] record_use skipped %s: %s",
                     account_id[:12], e)


async def state_for(
    db, *, user_id: str, account_id: str,
    identity_status: Optional[str] = None,
) -> dict:
    """The one answer, for one account.

    Precedence: last real use (if it is fresher than the cache window
    OR it is a hard failure) → the identity row → `connected`.
    """
    from app.db.models import AccountHealth

    out = {
        "account_state": "connected", "reason_code": "", "fix": "retry",
        "checked_at": None, "source": "default",
    }
    row = None
    try:
        from sqlalchemy import select
        row = (await db.execute(
            select(AccountHealth).where(
                AccountHealth.user_id == user_id,
                AccountHealth.account_id == account_id,
            )
        )).scalar_one_or_none()
    except Exception as e:  # noqa: BLE001
        logger.debug("[account_health] read skipped %s: %s",
                     account_id[:12], e)

    if row is not None and row.state:
        fresh = (
            row.checked_at is not None
            and row.checked_at > datetime.utcnow() - timedelta(days=7)
        )
        # A hard failure never goes stale on its own: nothing has said it
        # was fixed, so saying `Connected` again would be inventing news.
        if fresh or not is_transient(row.reason_code or ""):
            out.update({
                "account_state": row.state,
                "reason_code": row.reason_code or "",
                "fix": row.fix or "retry",
                "checked_at": (
                    row.checked_at.isoformat() + "Z"
                    if row.checked_at else None
                ),
                "source": row.source or "use",
            })
            if out["account_state"] != "connected":
                return out

    # No recorded use, or the recorded use was clean — fall back to the
    # identity row's own reading.
    mapped = {
        "active": ("connected", "", "retry"),
        "reauth_required": ("expired", "token_expired", "reconnect"),
        "revoked": ("revoked", "token_revoked", "reconnect"),
        "provider_down": ("connected", "vendor_down", "retry"),
    }.get((identity_status or "").lower())
    if mapped:
        out.update({
            "account_state": mapped[0], "reason_code": mapped[1],
            "fix": mapped[2], "source": "identity",
        })
    elif identity_status is not None:
        out.update({
            "account_state": "not_connected",
            "reason_code": "not_connected",
            "fix": "connect", "source": "identity",
        })
    return out
