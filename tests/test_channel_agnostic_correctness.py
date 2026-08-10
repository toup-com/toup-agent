"""Four places the core forked per-channel and one forgot a channel.

G-3..G-6 of docs/GAP_ANALYSIS.md (audit 2026-08-09). The theme: the platform
claims "one core pipeline, channels are thin adapters" — these tests pin the
four spots where that claim silently failed for specific channels.

1. Inbound-image tool gating fired only for the ws_chat caller (the only one
   that passes `inbound_attachments`). Telegram/WhatsApp deliver the same
   image as a media_path: the model could SEE it while analyze_image /
   edit_image stayed ungated — reproducing on every non-web channel exactly
   the "can't edit the image in this chat" bug with_inbound_image was
   written to fix.
2. `_channel_guidance` had no `whatsapp` entry, so a first-class channel
   (channel_util.KNOWN_CHANNELS) rendered "Unknown channel" guidance in the
   prompt prefix.
3. The vault tool's executor-side channel gate was a hardcoded COPY of the
   runner's VAULT_TOOL_CHANNEL_BLOCK and had already drifted (missing
   `autopilot`).
4. /v1/chat (+ SSE sibling) called AgentRunner.run with no channel at all →
   resolve_channel fell to "unknown".
"""

from __future__ import annotations

import inspect
import re


# ── 1. image gating derives from media_paths too ─────────────────────────

def test_image_gating_reads_media_paths_with_the_same_classifier_as_vision():
    """The gate and _build_media_content must share one view of 'is this an
    image' (mimetypes.guess_type) — a hardcoded extension list here would
    drift from what the model actually gets shown."""
    from app.agent.agent_runner import AgentRunner

    src = inspect.getsource(AgentRunner._run_inner)
    block = src.split("with_inbound_image(query_intent)")[0]
    assert "media_paths" in block and "guess_type" in block, (
        "inbound-image tool gating no longer considers media_paths — "
        "Telegram/WhatsApp images regress to 'can't edit this image'"
    )

    media_src = inspect.getsource(AgentRunner._build_media_content)
    assert "guess_type" in media_src, (
        "vision path stopped using mimetypes.guess_type; if this moves, move "
        "the gating classifier with it"
    )


def test_the_shared_classifier_gates_a_telegram_photo_and_not_a_document():
    import mimetypes

    def gates(path: str) -> bool:
        return (mimetypes.guess_type(path)[0] or "").startswith("image/")

    assert gates("/tmp/tg_download_abc.jpg")
    assert gates("/tmp/wa_media_1.png")
    assert not gates("/tmp/report.pdf")
    assert not gates("/tmp/notes.txt")


# ── 2. every first-class surface has guidance ────────────────────────────

def test_every_user_facing_channel_has_a_guidance_entry():
    """A first-class channel falling to the 'Unknown channel' default is a
    prefix-quality bug (it also contradicts channel_util's keep-in-sync
    contract).

    Read the real table, not its source text. This used to regex the dict
    literal out of ``AgentRunner``'s source, which meant the pin broke the
    moment the table was hoisted to module level (G-19b) even though every
    key it guards was still present — a source-shape assertion failing for
    a reason that has nothing to do with the invariant. The object is what
    the prompt builder actually reads, so a removed key still fails loudly,
    and a refactor that preserves behaviour no longer fails at all.
    """
    from app.agent.agent_runner import CHANNEL_GUIDANCE

    keys = set(CHANNEL_GUIDANCE)

    required = {"web", "app", "mobile", "voice", "telegram", "whatsapp",
                "discord", "slack", "extension", "vibecoding"}
    missing = required - keys
    assert not missing, f"channels with no guidance entry: {sorted(missing)}"


# ── 3. one vault channel-block set, not two ──────────────────────────────

def test_executor_vault_gate_uses_the_runners_set():
    from app.agent.agent_runner import VAULT_TOOL_CHANNEL_BLOCK
    from app.agent import tool_executor

    assert "autopilot" in VAULT_TOOL_CHANNEL_BLOCK, (
        "CP4.1 added autopilot; if this was removed on purpose, update the "
        "executor test below too"
    )

    src = inspect.getsource(tool_executor)
    handler = src.split("save_streaming_credential received unexpected")[0]
    tail = handler[-3000:]
    assert "VAULT_TOOL_CHANNEL_BLOCK" in tail, (
        "executor-side vault gate no longer imports the runner's set"
    )
    assert 'blocked = {"telegram"' not in src, (
        "a hardcoded copy of the vault channel block is back — it drifted "
        "once already (missing autopilot)"
    )


# ── 4. every AgentRunner.run call site in api_v1 names its channel ───────

def test_api_v1_run_calls_all_pass_a_channel():
    from app.api import api_v1

    src = inspect.getsource(api_v1)
    sites = re.findall(r"_agent_runner\.run\((.*?)\)\n", src, re.DOTALL)
    assert len(sites) >= 4, f"expected ≥4 run sites (2 chat, 2 voice), got {len(sites)}"
    unnamed = [s.strip()[:60] for s in sites if "channel=" not in s]
    assert not unnamed, (
        f"run() call sites with no channel= (resolve to 'unknown'): {unnamed}"
    )
