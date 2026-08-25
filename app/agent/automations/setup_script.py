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


def mode_label(mode: str, *, channel_label: str = "") -> str:
    """The §4.1 `mode_label`."""
    if mode == "posts":
        return f"posts to {channel_label}" if channel_label else "posts"
    return {"drafts_only": "drafts only", "reads_only": "reads only",
            "asks_first": "asks first"}.get(mode, mode)


def setup_turns(
    mode: str,
    *,
    channel_label: str = "",
    first_run_label: str = "tonight",
    scope_lines: list[dict] | None = None,
) -> list[dict]:
    """TurnDrafts for a fresh setup thread, in order.

    `scope_lines` — the capability check's step lines, engine-supplied:
    `[{"text": "Read new mail", "ok": True}, …]` with `ok=False` for a
    denied-by-design scope (rendered muted, §3.5 extension).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode!r}")
    opening = _OPENINGS[mode].format(channel=channel_label or "the channel you chose")
    if mode == "drafts_only" and first_run_label == "tonight":
        close = _CANVAS_DRAFTS_CLOSE
    else:
        close = f"First run is {first_run_label}. {_CLOSES[mode]}"
    return [
        {"kind": "agent", "text": opening},
        {
            "kind": "tool_request",
            "action": "Checked what I can do",
            "detail": mode_label(mode, channel_label=channel_label),
            "steps": list(scope_lines or []),
        },
        {"kind": "think", "text": _THINKS[mode]},
        {"kind": "agent", "text": close},
    ]
