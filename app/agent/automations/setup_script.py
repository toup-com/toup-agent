"""The mode-aware setup-thread script (R30 §5.3).

`from-template` and `describe` seed the `YOU ADDED THIS` thread with
these turns. The engine writes the note turn and assigns ids/seq; this
module supplies the TurnDrafts in emission order: the opening agent
line, the capability-check tool turn (whose step lines are the granted
and denied scopes — a denial is by design, muted, never danger), the
mode's think line, and the close. With `mode="drafts_only"` and
`first_run_label="tonight"` the strings reproduce the canvas fixture
byte-for-byte (`fixtures/automations/setup.json`); every other
combination uses the honest generic forms.

The canvas's `go ahead` user turn is a demo close and is not seeded.
"""

from __future__ import annotations

from app.services import automation_verbs as verbs

#: §4.1 mode → the label the wire serves ("posts" is completed with the
#: real target by the caller).
MODES = ("drafts_only", "reads_only", "posts", "asks_first")

_OPENINGS = {
    "drafts_only": "Here is what I will be able to do — write drafts, not send them.",
    "reads_only": (
        "Here is what I will be able to do — read, and tell you. "
        "I cannot change anything."
    ),
    "posts": (
        "Here is what I will be able to do — post one line in {channel}, "
        "nothing else."
    ),
    "asks_first": (
        "Here is what I will be able to do — prepare the change and wait "
        "for your yes before anything happens."
    ),
}

_THINKS = {
    "drafts_only": (
        "Less than it sounds like: I can create, not send. "
        "A mistake on my side stays a draft."
    ),
    "reads_only": (
        "Nothing I do here can change anything you own — reading is the whole of it."
    ),
    "posts": (
        "One line in one channel you chose. Anything more would need your yes first."
    ),
    "asks_first": "I stage the change and stop. Until you approve, nothing has happened.",
}

#: The close's second sentence per mode. The drafts/tonight pair keeps
#: the canvas bytes; everything else says when honestly.
_CLOSES = {
    "drafts_only": "The drafts will be waiting, and every step will be here.",
    "reads_only": "The brief will be waiting, and every step will be here.",
    "posts": "The post will be waiting for your edit, and every step will be here.",
    "asks_first": (
        "The change will be staged and waiting on you, and every step will be here."
    ),
}

_CANVAS_DRAFTS_CLOSE = (
    "First run is tonight. In the morning the drafts will be waiting, "
    "and every step will be here."
)


def _channel_only(channel_label: str) -> str:
    """The channel, whether the caller handed us the channel or the
    mode label that contains it. `mode_of` builds `"posts to {label}"`,
    and both call sites pass that whole string."""
    text = (channel_label or "").strip()
    lowered = text.lower()
    for prefix in ("posts to ", "post to "):
        if lowered.startswith(prefix):
            return text[len(prefix):].strip()
    # `mode_of` returns the bare word "posts" when the write target has
    # no label, and a caller passing that mode label would otherwise
    # render "post one line in posts, nothing else." The fallback
    # ("the channel you chose") is what that case is for.
    if lowered in ("posts", "post"):
        return ""
    return text


def scope_lines_from(
    permissions: dict, *, connector_name: str = "",
) -> list[dict]:
    """`permissions.resolve`'s `{can, cant}` → the capability check's
    step lines (§5.3).

    Both call sites passed `[]` here, so the one turn in a new
    automation's thread whose whole job is to say what it will and will
    not be able to do said nothing at all — the setup script's most
    load-bearing turn, empty, on every automation ever created.

    A denial is rendered muted, never as danger: "it cannot send mail"
    is the reassurance, not the warning.
    """
    def _line(entry: dict, ok: bool) -> dict:
        label = str((entry or {}).get("label") or "").strip()
        if connector_name and label:
            label = f"{label} · {connector_name}"
        return {"text": label, "ok": ok}

    lines = [_line(p, True) for p in (permissions or {}).get("can") or []]
    lines += [_line(p, False) for p in (permissions or {}).get("cant") or []]
    return [line for line in lines if line["text"]]


def writer_connectors(raw: dict) -> set:
    """Connector ids whose step (v2) or action (v1) WRITES, from a raw
    spec dict. R38 — the per-account verb's ground truth: only these
    accounts wear the write-mode label on their capability check."""
    from app.services.automation_verbs import is_write_tool
    out = {
        s.get("connector_id")
        for s in (raw.get("steps") or [])
        if isinstance(s, dict)
        and (s.get("grant_id") or is_write_tool(s.get("tool")))
    }
    action = raw.get("action") or {}
    if isinstance(action, dict) and (
        action.get("grant_id") or is_write_tool(action.get("tool"))
    ):
        out.add(action.get("connector_id"))
    out.discard(None)
    return out


def mode_label(mode: str, *, channel_label: str = "") -> str:
    """The §4.1 `mode_label`."""
    if mode == "posts":
        return f"posts to {channel_label}" if channel_label else "posts"
    return {"drafts_only": "drafts only", "reads_only": "reads only",
            "asks_first": "asks first"}.get(mode, mode)


def setup_turns(
    mode: str,
    channel_label: str = "",
    first_run_label: str = "tonight",
    scope_lines: list[dict] | None = None,
    *,
    accounts: list[dict] | None = None,
    blocked: bool = False,
    format_noun: str = "a ranked list",
) -> list[dict]:
    """TurnDrafts for a fresh setup thread, in order.

    Positional-tolerant on purpose: the from-template endpoint calls it
    positionally. `channel_label` is only read by `posts` mode — and
    the claim that once stood here, that a call site passing the MODE
    label there is harmless, was false for exactly that mode. Both call
    sites do `mode, label = mode_of(...)` and pass `label`, which for
    `posts` is already `"posts to #all-toup"`. The opening then read
    "post one line in posts to #all-toup, nothing else." and the
    capability check's detail read "posts to posts to #all-toup" — in
    the first four turns of a new automation's thread, which is the
    first thing anyone reads about it. `_channel_only` now strips the
    prefix whatever the caller passes, so the mistake cannot reach a
    user from any call site.

    A `first_run_label` carrying a capitalised schedule sentence
    ("Weekdays at 8:00") is lowered so the close reads as one sentence.
    `scope_lines` — the capability check's step lines, engine-supplied:
    `[{"text": "Read new mail", "ok": True}, …]` with `ok=False` for a
    denied-by-design scope (rendered muted, §3.5 extension).

    `accounts` — round 35: `[{"account_id": cid, "steps": [...]}, …]`,
    ONE capability turn per account the automation uses. The single
    flattened turn was stamped `members[0]` at both call sites, so a
    six-account Morning work brief opened with "Checked 1 account" and
    a lone Jira chip — the founder read that as "it only looked at
    Jira", which is exactly what the card said. When `accounts` is
    given it wins over `scope_lines`; the legacy single-turn shape
    survives for a caller that has nothing better.

    R38: an entry may carry `"writes": bool` — whether THAT account has
    a write step. The automation-level mode was stamped as every
    account's detail, so a posts-to-Slack brief showed Gmail and
    Outlook sub-labelled "posts" while their own drill-in said Read
    new mail / ✕ Send anything, and the ⋯ menu said "reads only" one
    screen away (rec1 f007–f018). With the flag: only the writing
    account wears the write-mode label; a reads-only account says
    "reads only". Entries without the flag keep the legacy stamp.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    channel_label = _channel_only(channel_label)
    opening = _OPENINGS[mode].format(channel=channel_label or "the channel you chose")
    label = (first_run_label or "tonight").strip() or "tonight"
    if label[:1].isupper() and not label.startswith("I "):
        label = label[:1].lower() + label[1:]
    if blocked:
        # R39 (founder P6): never promise "First run is soon" about an
        # automation `workflow.run_blockers` says cannot fire — that
        # close stood in the same thread as run-now's needs_setup 409.
        # The blocker itself was already asked for two bubbles up.
        close = ("It runs the moment that is picked, and every step "
                 "will be here.")
    elif mode == "drafts_only" and label == "tonight":
        close = _CANVAS_DRAFTS_CLOSE
    else:
        close = f"First run is {label}. {_CLOSES[mode]}"
    detail = mode_label(mode, channel_label=channel_label)
    if accounts:
        checks = [
            {
                "kind": "tool",
                "account_id": str(a.get("account_id") or ""),
                "action": "Checked what I can do",
                "detail": (
                    detail if a.get("writes", True)
                    else mode_label("reads_only")
                ),
                "steps": list(a.get("steps") or []),
            }
            for a in accounts
            if a.get("account_id")
        ]
    else:
        checks = [{
            "kind": "tool",
            "action": "Checked what I can do",
            "detail": detail,
            "steps": list(scope_lines or []),
        }]
    return [
        {"kind": "agent", "text": opening},
        *checks,
        *_band_turns(accounts),
        {"kind": "think", "text": _THINKS[mode]},
        {"kind": "agent", "text": close},
        *_shape_turn(accounts, format_noun),
    ]


def _band_turns(accounts: list[dict] | None) -> list[dict]:
    """One line per BAND, never per account (spec §8).

    The narration used to be one line per connector, so a six-account
    automation opened with six near-identical "Connected X" bubbles
    before it said anything about itself. The band is the grain a person
    reads at: mail, channels, tickets, plans.
    """
    from .build_ledger import band_of, _BANDS, _join
    if not accounts:
        return []
    members: dict[str, list[str]] = {}
    writes: dict[str, bool] = {}
    for a in accounts:
        cid = str(a.get("account_id") or "")
        if not cid:
            continue
        key = band_of(cid)
        members.setdefault(key, []).append(cid)
        writes[key] = writes.get(key, False) or bool(a.get("writes"))
    order = [k for k, _m, _p in _BANDS] + ["MORE"]
    out: list[dict] = []
    for key in order:
        cids = members.get(key)
        if not cids:
            continue
        names = _join([verbs.display_name(c) or c for c in cids])
        # A band where any member can write must not claim read-only for
        # the whole band; the per-account sheet is where the detail lives.
        sub = ("read and write where you allowed it, scoped to you"
               if writes.get(key) else "read only, scoped to you")
        out.append({"kind": "agent", "text": f"Connected {names} — {sub}"})
    return out


def _shape_turn(
    accounts: list[dict] | None, format_noun: str,
) -> list[dict]:
    """The sentence that closes the setup (spec §8).

    Skipped when there is nothing to name: a shape sentence about no
    accounts is a sentence about nothing.
    """
    from .build_ledger import band_of, closing_sentence, _BANDS
    if not accounts:
        return []
    members: dict[str, list[str]] = {}
    for a in accounts:
        cid = str(a.get("account_id") or "")
        if cid:
            members.setdefault(band_of(cid), []).append(cid)
    order = [k for k, _m, _p in _BANDS] + ["MORE"]
    band_order = [(k, members[k]) for k in order if members.get(k)]
    if not band_order:
        return []
    return [{"kind": "agent",
             "text": closing_sentence(band_order, format_noun)}]
