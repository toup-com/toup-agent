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
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    channel_label = _channel_only(channel_label)
    opening = _OPENINGS[mode].format(channel=channel_label or "the channel you chose")
    label = (first_run_label or "tonight").strip() or "tonight"
    if label[:1].isupper() and not label.startswith("I "):
        label = label[:1].lower() + label[1:]
    if mode == "drafts_only" and label == "tonight":
        close = _CANVAS_DRAFTS_CLOSE
    else:
        close = f"First run is {label}. {_CLOSES[mode]}"
    return [
        {"kind": "agent", "text": opening},
        {
            "kind": "tool",
            "action": "Checked what I can do",
            "detail": mode_label(mode, channel_label=channel_label),
            "steps": list(scope_lines or []),
        },
        {"kind": "think", "text": _THINKS[mode]},
        {"kind": "agent", "text": close},
    ]
