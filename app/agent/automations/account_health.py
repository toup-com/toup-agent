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
    # ── "I do not know why" is not "it is fine" (round 33, item 4) ──────
    # Every unclassified failure used to alias to `timeout`, which is a
    # TRANSIENT state — so the card said "Connected" with a "Try again"
    # button for a 401, a 403 org policy and a missing scope alike. An
    # unknown failure keeps the account out of `connected` and offers the
    # PROBE, which is the only honest next step: ask the vendor.
    "unknown_error": ("needs_check", "check"),
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
    # `unreachable` is what `_failure_reason` answers for everything it
    # does not recognise — a real 401/403/404 included — so aliasing it to
    # a transient `timeout` told the user their broken connector was
    # having a bad minute. It is now what it says: unknown.
    "unreachable": "unknown_error",
    "tool_error": "unknown_error",
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
# The dispatcher's own grant-gate refusal (connector_dispatcher.
# _resolve_automation_grant). R36-6: this is a permission the user was
# never asked for, not a connector that broke — it must never render as
# "Could not reach {Connector}" / "it did not tell me why", which is
# exactly what it did when a template's ungranted draft step reached
# the gate. Matched BEFORE the scope regex: the gate's wording contains
# "permission", and scope_missing's "grant more access" button is the
# wrong fix for a grant that needs approving in the thread.
_GRANT_GATE_RE = re.compile(
    r"no approved permission backs this automation call"
    r"|automations fail closed"
    r"|permission backing this automation",
    re.IGNORECASE,
)

PROBE_CACHE_S = 600  # ten minutes — vendor rate limits, §4.4


# ── the string table ─────────────────────────────────────────────────

_TABLE: Optional[dict] = None


def _table_path() -> str:
    """The table SHIPPED WITH THE PACKAGE (round 33, item 4).

    This used to resolve four directories above this module — the repo
    root locally, and `/` inside the agent image, because
    `.github/workflows/build-agent.yml` builds with `context: ./backend`
    and `fixtures/` lives at the repo root. So in production the table
    was never found, `strings()` answered `{}`, and EVERY sentence and
    EVERY button label on every connector card fell through to its
    hardcoded fallback: "I could not read Jira." with a "Try again"
    button, for a token that had expired and needed reconnecting.
    Four cards, four different real reasons, one wrong sentence.

    `copy_guard.py` already said this in writing — "a deployed image
    that ships no fixtures directory" — and solved it by embedding its
    data in the module. This does the same thing with a file, because
    the table is 18KB of copy: it lives in the package, and
    `fixtures/automations/reason-strings.json` stays the contract
    document, held byte-identical by
    `test_automation_reason_strings_ship_with_the_package`.
    """
    # Beside the module, NOT in a `data/` subdirectory: `.gitignore` has a
    # bare `data/` rule, so a copy filed there is never committed and never
    # reaches the image — the same defect this function is fixing, one
    # directory deeper.
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "reason_strings.json")


def strings() -> dict:
    """C's `account_state × surface` table. Loaded once.

    An unreadable table is not survivable — every card, every button
    label and every honest line degrades to a fallback that says the
    wrong thing confidently — so it is logged at ERROR. It still does
    not RAISE: a run that dies because a copy file moved is worse than
    a run that reports plainly.
    """
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    try:
        with open(_table_path(), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        _TABLE = data if isinstance(data, dict) else {}
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.error(
            "[account_health] reason-strings UNREADABLE at %s: %s — every "
            "connector card will fall back to generic copy",
            _table_path(), e,
        )
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
    if _GRANT_GATE_RE.search(message or ""):
        return "grant_missing"
    if _ORG_APPROVAL_RE.search(message or ""):
        return "org_approval_needed"
    if _SCOPE_RE.search(message or ""):
        return "scope_missing"
    # ── Read the provider's own status before giving up (round 33) ──────
    # The RPC envelope kind is the engine's word for what happened, and
    # for anything it does not model it says `tool_error`/`unreachable`.
    # The provider's message still carries the real answer — a 401, an
    # `invalid_grant`, a 429 — and reading it here is the difference
    # between "reconnect Gmail" and a "Try again" button on an account
    # that will never answer until it is reconnected.
    if token not in _LEGACY_REASON_ALIASES and token in _REASON_MAP:
        return token
    hit = _PROVIDER_STATUS_RE.search(message or "")
    if hit:
        return hit.group(0) and _status_reason(hit)
    return _LEGACY_REASON_ALIASES.get(token, token) or "timeout"


# The provider signals worth recognising by name. Narrow on purpose: a
# wrong confident code is worse than `unknown_error`, which at least
# offers the probe.
_PROVIDER_STATUS_RE = re.compile(
    r"\b(invalid_grant|unauthorized|unauthenticated|401"
    r"|rate.?limit|too many requests|429"
    r"|50[0234]|bad gateway|service unavailable|gateway timeout"
    r"|timed? ?out)\b",
    re.IGNORECASE,
)


def _status_reason(hit) -> str:
    text = hit.group(0).lower()
    if text in ("invalid_grant", "unauthorized", "unauthenticated", "401"):
        return "token_expired"
    if "429" in text or "rate" in text or "many requests" in text:
        return "rate_limited"
    if text.startswith("50") or text in ("bad gateway", "service unavailable",
                                         "gateway timeout"):
        return "vendor_down"
    return "timeout"


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


def fix_for(state: str, reason_code: str) -> str:
    """The remedy an account offers, from §4.4 and nothing else.

    A CONNECTED account offers one only when its last failure was
    transient (rate_limited | vendor_down | timeout keep the state
    `connected` and get `fix: retry`). Absent a transient reason there is
    nothing to fix — and the app renders ANY non-empty fix, so a blanket
    `retry` default drew "Try again" on an account that was simply
    working. R31-13's own class, inverted.

    `is_transient` reads the string table, so this rule and the table can
    never drift.
    """
    if state == "connected":
        return "retry" if is_transient(reason_code or "") else ""
    return "reconnect"


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
    ) or fix_button(fix, connector_id, display)
    # ── A label may never contradict the fix it runs (round 33, item 4) ──
    # These two lines used to end `or "Try again"` and
    # `or f"I could not read {display}."`. With the table unreadable in
    # production (see `_table_path`) they fired for EVERY reason code, so
    # an expired token wore a "Try again" button that retried a read that
    # could not succeed, over a sentence that named no cause. The label now
    # comes from the fix or from nowhere: the app renders no button for an
    # empty label, and no button is honest where a wrong one is not.
    if not label and fix:
        logger.error(
            "[account_health] no button label for fix=%s reason=%s — the "
            "card will render without one", fix, reason_code,
        )
    if not sentence:
        # The reason's own words, at minimum. `reason_block` is the row
        # `sentence_for` failed to render a surface from; its `sheet_detail`
        # is the short cause ("access expired"), which is strictly more than
        # a bare "I could not read X."
        detail = sheet_detail(reason_code)
        sentence = (f"I could not read {display} — {detail}." if detail
                    else f"I could not read {display}, and I do not yet "
                         f"know why.")
    return {
        "account_id": account_id,
        "connector_id": connector_id,
        "name": display,
        "reason_code": reason_code,
        "sentence": sentence,
        "fix": fix,
        "fix_label": label,
        "approval_url": approval_url,
    }


async def record_use(
    db, *, user_id: str, account_id: str, ok: bool,
    reason_code: str = "", message: str = "", broadcast: bool = True,
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
    # A successful call clears the account, and a cleared account offers
    # NO remedy — `fix_for("connected", "")` is "". Writing `retry` here
    # is how a healthy row still produced a "Try again" button after the
    # two readers were fixed: `state_for` prefers the stored `row.fix`,
    # so a wrong value written once outlives every read-side correction.
    account_state, fix = (
        ("connected", fix_for("connected", "")) if ok
        else state_for_reason(code)
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
        changed = (row.state != account_state or row.reason_code != code)
        row.state = account_state
        row.reason_code = code
        row.fix = fix
        row.checked_at = now
        row.source = "use"
        await db.flush()
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.debug("[account_health] record_use skipped %s: %s",
                     account_id[:12], e)
        return
    # ── One state, every surface (round 33, item 4) ─────────────────────
    # `AccountHealth` is the record of what a REAL call did, and until now
    # only the account sheet read it — the thread card, the Connectors page
    # and the drawer each derived their own answer, so a run could fail to
    # read Jira while every other surface still said "Connected". The
    # `connector.state` frame is the app's one ingress for this
    # (`store.connectors`, via the app-level bridge), and it was emitted
    # only on OAuth connect/expire hooks: a failed READ broadcast nothing.
    # Now it does — on a CHANGE, so a run that reads four accounts every
    # morning does not narrate four unchanged states every morning.
    #
    # `broadcast=False` is for the two OAuth hooks, which record the use
    # and then emit their OWN frame a few lines later: without it the app
    # got the same `connector.state` twice for one event, and the second
    # is not free — every frame re-renders the Connectors page and the
    # drawer count.
    if not changed or not broadcast:
        return
    try:
        from .connector_state import emit_state_frame
        await emit_state_frame(
            user_id, connector_id=account_id, state=account_state,
            reason_code=code, fix=fix,
            reconnected_at=now if ok else None,
        )
    except Exception as e:  # noqa: BLE001 — a frame never fails a run
        logger.debug("[account_health] state frame skipped %s: %s",
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
        # No recorded use and no identity reading: healthy, and a healthy
        # account offers no remedy (`fix_for`).
        "account_state": "connected", "reason_code": "", "fix": "",
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
                "fix": row.fix or fix_for(row.state, row.reason_code or ""),
                "checked_at": (
                    row.checked_at.isoformat() + "Z"
                    if row.checked_at else None
                ),
                "source": row.source or "use",
            })
            if out["account_state"] != "connected":
                return out
            # Connected, but the recorded use still NAMED something — a
            # transient failure keeps the state and keeps its reason. The
            # identity row below knows only `active`, so falling through
            # would overwrite `timeout` with "" and drop the retry with
            # it. That erasure was invisible while every connected
            # account defaulted to `retry` anyway: the fix survived and
            # only the reason was lost, so the sheet said an account was
            # fine and offered a remedy for nothing.
            if out["reason_code"]:
                return out

    # No recorded use, or the recorded use was clean — fall back to the
    # identity row's own reading.
    mapped = {
        # `active` is a healthy account: no reason, and so no remedy.
        # `provider_down` below is transient, which is why IT keeps retry.
        "active": ("connected", "", ""),
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


# ── the scope probe ──────────────────────────────────────────────────
#
# The scopes an automation's READS actually need, per connector. These
# are the four §4.4 names and nothing more: a probe that enumerated
# every scope a connector could want would fail accounts that work.
#
# `scopes_read` has existed in the manifests since R26 and had NO
# consumer — three references in the whole repo, all of them printing
# it. So "does this connection hold what this automation needs" was
# never asked anywhere except reactively, from a provider's 403, which
# is why an Outlook connection with no `Mail.Read` presented as
# `IT CAN Read new mail` right up until the run failed.
REQUIRED_READ_SCOPES: dict[str, tuple[str, ...]] = {
    "outlook": ("Mail.Read",),
    "slack": ("channels:history", "groups:history"),
    "github": ("repo",),
    "teams": ("Chat.Read",),
}

# Any ONE of these is enough (Slack's two history scopes cover public
# and private channels; an account with either can read something).
_ANY_OF: frozenset = frozenset({"slack"})


def missing_scopes(connector_id: str, granted: list) -> list[str]:
    """Which required read scopes this grant does not hold."""
    need = REQUIRED_READ_SCOPES.get(connector_id) or ()
    if not need:
        return []
    have = {str(s).strip() for s in (granted or []) if str(s).strip()}
    if not have:
        # `scopes_json` empty means the provider did not tell us. Not
        # knowing is not the same as knowing it is missing — claiming a
        # scope problem here would send a working account to a consent
        # screen it does not need.
        return []
    missing = [s for s in need if s not in have]
    if connector_id in _ANY_OF and len(missing) < len(need):
        return []
    return missing


async def probe(
    db, *, user_id: str, account_id: str, force: bool = False,
) -> dict:
    """Ask what this account can actually do, and record the answer.

    Order matters. The identity's own status is checked first because a
    dead token makes every other question moot; the scope check runs
    only on a live one. A probe never invents a failure: if we cannot
    tell, the recorded state is unchanged.
    """
    from . import registry as reg

    cached = await state_for(db, user_id=user_id, account_id=account_id)
    if not force and cached.get("checked_at"):
        try:
            seen = datetime.fromisoformat(
                str(cached["checked_at"]).rstrip("Z"))
            if (datetime.utcnow() - seen).total_seconds() < PROBE_CACHE_S:
                return {**cached, "cached": True}
        except (ValueError, TypeError):
            pass

    state, reason, fix = "connected", "", "retry"
    granted: list = []
    try:
        conn = (await reg.fetch_connection_state(user_id)).get(
            account_id) or {}
        granted = list(conn.get("scopes") or [])
        status = (conn.get("status") or "").lower()
        if not conn:
            state, reason, fix = "not_connected", "not_connected", "connect"
        elif status == "revoked":
            state, reason, fix = "revoked", "token_revoked", "reconnect"
        elif status == "reauth_required":
            state, reason, fix = "expired", "token_expired", "reconnect"
        elif status == "provider_down":
            # Transient: the credential is fine, the vendor is not.
            state, reason, fix = "connected", "vendor_down", "retry"
        elif not conn.get("connected"):
            state, reason, fix = "not_connected", "not_connected", "connect"
        else:
            gap = missing_scopes(account_id, granted)
            if gap:
                state = "scope_missing"
                reason = f"scope_missing:{gap[0]}"
                fix = "grant"
    except Exception as e:  # noqa: BLE001 — a probe never invents news
        logger.warning("[account_health] probe failed %s: %s",
                       account_id[:12], e)
        return {**cached, "cached": False, "probed": False}

    # An org-approval refusal is only visible from a real call, and this
    # probe does not make one — so a standing `org_approval_needed`
    # SURVIVES a clean scope probe. Clearing it here would tell the user
    # their owner had approved when nothing had happened.
    if cached.get("reason_code") == "org_approval_needed" \
            and state == "connected":
        return {**cached, "cached": False, "probed": True}

    now = datetime.utcnow()
    try:
        from sqlalchemy import select
        from app.db.models import AccountHealth
        row = (await db.execute(
            select(AccountHealth).where(
                AccountHealth.user_id == user_id,
                AccountHealth.account_id == account_id,
            )
        )).scalar_one_or_none()
        if row is None:
            row = AccountHealth(user_id=user_id, account_id=account_id)
            db.add(row)
        row.state = state
        row.reason_code = reason
        row.fix = fix
        row.checked_at = now
        row.source = "probe"
        row.scopes_json = json.dumps(granted, default=str)[:4000]
        await db.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("[account_health] probe write skipped %s: %s",
                       account_id[:12], e)

    out = {
        "account_state": state, "reason_code": reason, "fix": fix,
        "checked_at": now.isoformat() + "Z", "source": "probe",
        "cached": False, "probed": True,
    }
    try:
        from .connector_state import emit_state_frame
        await emit_state_frame(
            user_id, connector_id=account_id, state=state,
            reason_code=reason, fix=fix,
        )
    except Exception as e:  # noqa: BLE001 — no live socket is normal
        logger.debug("[account_health] probe frame skipped: %s", e)
    return out
