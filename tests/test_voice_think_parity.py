"""Voice `think` parity: the in-process path must not keep the defect the
V2 relay path fixed (G-8), and the #488 date guard must actually ship (G-9).

G-8 — `ws_realtime`'s in-process Option-A `think` called AgentRunner.run
with no save flags (defaults True). Its HTTP siblings send `save: False`
(ws_realtime.py `"save": False`), which api_v1's agent-turn maps to
save_user_message=False / save_assistant_message=False /
disable_post_processing=True, with a documented incident behind it: `task`
is a string the REALTIME MODEL synthesised, not what the user said — five
bogus 409A memories were minted from it, and _save_voice_messages persists
the real transcripts anyway, so saving `task` double-writes the day chat.
The in-process path is latent on the current platform-relay topology and
live in monolith mode; latent is not fixed.

G-9 — voice_day_context_date_guard shipped dark 2026-07-31 "pending a
canary listen" that never happened; the wrong-day header (#488) stayed
live by default for nine days. Default flipped ON 2026-08-09 (label-only
change, no context dropped; escape hatch env var remains).
"""

from __future__ import annotations

import inspect
import re


def test_in_process_think_passes_the_v2_save_semantics():
    from app.api import ws_realtime

    src = inspect.getsource(ws_realtime)
    # The Option-A in-process call is the only `_agent_runner.run(` site.
    sites = re.findall(r"_agent_runner\.run\((.*?)\n\s*\)", src, re.DOTALL)
    assert len(sites) == 1, f"expected exactly one in-process run site, got {len(sites)}"
    site = sites[0]
    for kwarg in ("save_user_message=False",
                  "save_assistant_message=False",
                  "disable_post_processing=True"):
        assert kwarg in site, (
            f"in-process think lost `{kwarg}` — it would persist the "
            "realtime model's synthesised task next to the real transcripts "
            "and mine memories from words the user never said"
        )


def test_the_relay_paths_still_send_save_false():
    """The parity target: if the relay ever changes its mind, this test and
    the one above must change together, not drift apart."""
    from app.api import ws_realtime

    src = inspect.getsource(ws_realtime)
    assert '"save": False' in src, (
        "the V2 relay no longer sends save=False — re-derive the in-process "
        "flags from whatever it does now"
    )


def test_date_guard_defaults_on():
    from app.config import settings

    assert settings.model_fields["voice_day_context_date_guard"].default is True, (
        "the #488 wrong-day-header fix went dark again — flipping this back "
        "off re-labels a previous day's transcript as 'today' on the "
        "ordinary voice-first morning"
    )
