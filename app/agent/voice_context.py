"""One context assembler — the AGENT builds voice's instructions (G-19a PR-A).

Until now the Realtime session's instructions were assembled on
platform-api by `app.api.ws_realtime.build_realtime_instructions`, from a
DIFFERENT database than the one text chat speaks from, through a
hand-copy of the runner's persona renderer. Nine divergences were
catalogued; four of them are ended here, by construction:

  D1  identity ORDER — voice sorted by priority and appended, so a
      `user_profile` at 90 rendered ABOVE a `soul` at 50. Both sides now
      call `agent_runner.render_identity_sections`, whose rule is
      priority-desc THEN soul hoisted to the front.
  D3  the no-soul fallback — voice only emitted a (blander) default when
      the ENTIRE prefetch produced nothing, so a user with memories but
      no soul document got NO persona at all on voice while text chat got
      the personal-agent default. One renderer, one trigger.
  D4  day SELECTION — voice asked for `/api/day-chats?limit=1`, which is
      ordered `local_date DESC`, and printed whatever came back under a
      "Today's ..." header. On a voice-first morning that is YESTERDAY
      (#488). The live relay closed #488 with a DATE GUARD
      (`voice_day_context_date_guard`): it still serves the newest
      existing day, but when that day is not the user's local today the
      header names the real date and says nothing has been said today.

      This module reproduces that shipped behaviour BYTE FOR BYTE —
      newest existing day chat, the guard's exact header — rather than
      the earlier draft's resolve-today-and-create approach. The W-6
      flip criterion is `ctx_shadow match=True` on consecutive real
      sessions, and match is computed over section fingerprints; a day
      block that is *better* than the relay's is still a mismatch, and a
      mismatch withholds the flip. #488 stays fixed the way users
      actually have it today: the transcript is never mislabelled as
      today's. The zone that decides "is this today" is validated first
      (`_resolve_effective_tz`, relay-passed zone falling back to the
      tenant's `User.timezone`); with no resolvable zone the header
      carries no date claim at all, exactly like the relay with the
      guard unresolved, and `day_timezone` is reported in `degraded`.
  D5  data plane — voice read `identities` from the PLATFORM DB (a
      partitioning leftover: `identities` is AGENT_ONLY, see
      tests/test_agent_serves_identity.py) and memories over HTTP. This
      runs INSIDE the tenant, on the tenant's own session — the same rows
      text chat reads.

Deliberately NOT changed here (see the module docstring of ws_realtime
for the legacy copies, which stay until a live canary):
  D7  memory rendering keeps voice's shape — an unranked dump of both
      brains with head-caps — rather than the runner's hybrid retrieval.
      Switching retrieval strategy is a product change; this PR is a
      relocation.
  D9  `# Voice — Always Apply` (the runner's tone guardrails) has no
      voice counterpart and does not get one here.

PR-A wires nothing: `POST /v1/internal/voice-context` exposes this on the
agent container and the relay still uses its own builder. The relay swap
is PR-B, behind a canary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.agent_runner import (
    render_identity_anchor,
    render_identity_sections,
)

logger = logging.getLogger(__name__)


# Ceiling the agent's own /api/memories enforces (le=200). Kept identical
# so the dump this module produces is the same set the relay used to
# fetch over HTTP.
VOICE_MEMORIES_LIMIT = 200

# The relay requests `/{date}/messages?limit=500`; the same cap here keeps
# a 501-message day rendering the same 500 rows on both sides.
DAY_MESSAGES_LIMIT = 500

# Budget split, unchanged from the relay: agent brain keeps its head
# (highest-priority entries first), user brain likewise, day history keeps
# its TAIL (the newest turns are the ones that matter in a live call).
BUDGET_SHARE_AGENT_BRAIN = 0.2
BUDGET_SHARE_USER_BRAIN = 0.3
BUDGET_SHARE_DAY_HISTORY = 0.5

# Section keys, in the order they are joined into `instructions`.
# `identity_anchor` sits at index 1 — the runner's position
# (prompt_profile._FULL_SECTIONS), NOT the relay's, which appended it
# after the whole day transcript. Drift D2: a white-label guard that the
# model reads 20k characters after the persona is a guard the model has
# already contradicted.
VOICE_SECTION_ORDER = (
    "identity",
    "identity_anchor",
    "agent_brain",
    "user_brain",
    "day_history",
    "voice_mode",
    "onboarding",
)


@dataclass
class VoiceContext:
    """Everything a Realtime session needs, assembled once.

    `sections` carries the same blocks `instructions` was joined from, so
    a caller can re-budget or drop one without re-deriving it, and a test
    can pin a single block instead of a 20k-character string.
    `degraded` names legs that FAILED — a total context failure must never
    look the same as a user who simply has no data (the 2026-07-31
    incident: every voice session ran with no persona and no brain, and
    the prompt looked exactly like a new user's).

    `empty` names legs that succeeded and legitimately had nothing. The
    split is the whole point: a first voice call of the morning has no
    day transcript yet, and if that raised the same alarm as a failed
    query, the alarm would fire for every user every day and operators
    would learn to ignore it — which is how the 2026-07-31 incident
    stayed invisible in the first place.
    """

    instructions: str = ""
    day_date: Optional[str] = None
    sections: Dict[str, str] = field(default_factory=dict)
    degraded: List[str] = field(default_factory=list)
    empty: List[str] = field(default_factory=list)


def cap_chars(text: str, max_chars: int, keep: str = "head") -> str:
    """Trim a section to a budget on line boundaries.

    keep="head" preserves the start (memory lists: highest-priority
    entries come first); keep="tail" preserves the end (day history:
    newest messages last). Byte-for-byte the relay's `_cap_chars`; the
    relay keeps its copy until the legacy builder is deleted.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    lines = text.split("\n")
    header, body = lines[0], lines[1:]
    kept: List[str] = []
    used = len(header)
    seq = body if keep == "head" else reversed(body)
    for ln in seq:
        if used + len(ln) + 1 > max_chars:
            break
        kept.append(ln)
        used += len(ln) + 1
    if keep == "tail":
        kept.reverse()
    marker = "- [context trimmed to budget]"
    return "\n".join([header, marker] + kept) if keep == "tail" else "\n".join([header] + kept + [marker])


def _render_brain(header: str, memories: List[Any]) -> str:
    """`# Header` + one `- [category] content` row per memory."""
    lines = [header]
    for m in memories:
        cat = getattr(m, "category", "") or ""
        content = getattr(m, "content", "") or ""
        lines.append(f"- [{cat}] {content}")
    return "\n".join(lines)


def day_history_header(
    total: int, day_date: Optional[str], local_today: Optional[str]
) -> str:
    """Header for the day block — the relay's `_day_history_header`, byte
    for byte (a cross-pin test holds the two copies together until the
    legacy builder is deleted and this one remains).

    `local_today` is None when the zone could not be resolved — then the
    historical wording is returned with no date claim, exactly like the
    relay with its guard unresolved. With a known local date and a block
    that is genuinely not from today, the block is still included but
    labelled with its real date, because the Realtime model otherwise
    narrates a previous day as "earlier today" (#488).
    """
    _today = f"# Today's Full Conversation History ({total} messages across all channels)"
    if not local_today or not day_date or day_date == local_today:
        return _today
    return (
        f"# Conversation from {day_date} — the last day you and the user "
        f"spoke ({total} messages across all channels). This is NOT today: "
        f"today is {local_today} and nothing has been said today yet. Do "
        f"not describe any of it as having happened today."
    )


def render_day_history(
    rows: List[Any], day_date: Optional[str], local_today: Optional[str]
) -> str:
    """The day block, byte-identical to the relay's rendering.

    `rows` are (role, content, channel) in the exact shape and order
    `GET /api/day-chats/{date}/messages` serves (raw content, channel
    already defaulted to "web"). Two relay quirks are reproduced on
    purpose, because the shadow comparator hashes bytes:

      * the header counts the RAW rows (`total = len(day_msgs)`), while
        the lines below filter to user/assistant turns with non-empty
        content — so a day containing tool rows shows a count larger
        than its rendered lines, on both sides identically;
      * each line is `{speaker} [{channel}]: {content}` — the relay's
        format, NOT `load_day_context`'s `[{channel} {local time}]`
        annotation. Richer labelling is a product change to make ONCE,
        in this file, after the legacy copy is gone.
    """
    if not rows:
        return ""
    lines = [day_history_header(len(rows), day_date, local_today)]
    for role, content, channel in rows:
        content = (content or "").strip()
        if role in ("user", "assistant") and content:
            speaker = "User" if role == "user" else "You"
            ch = f" [{channel}]" if channel else ""
            lines.append(f"{speaker}{ch}: {content}")
    if len(lines) <= 1:
        return ""
    return "\n".join(lines)


def render_voice_mode(now_utc: datetime) -> str:
    """The speech-format / tool-policy block.

    A relocation of ws_realtime's `# Voice Conversation Mode`, with the
    `think` paragraph fixed at the V2 wording: this assembler runs on the
    agent container, which is the very thing `think` hops to — the V1
    "reasoning model only, no tools" caveat is false wherever this code
    can execute. Kept out of `render_identity_*` on purpose: this is a
    CHANNEL document, not a persona one.
    """
    now_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    return (
        "# Voice Conversation Mode\n"
        "You are in a LIVE VOICE conversation. Follow these rules:\n"
        "- Respond naturally and conversationally, as if speaking face-to-face.\n"
        "- Keep responses concise — aim for 1-3 sentences unless the user asks for detail.\n"
        "- Do NOT use markdown, code blocks, bullet points, or any text formatting.\n"
        "- Do NOT say 'here is a list' or read structured data verbatim.\n"
        "- Use natural speech patterns: contractions, casual phrasing.\n"
        "- Match the user's language. Speak EVERY language with a natural, NATIVE "
        "accent and native pronunciation — never a foreign or English-accented one.\n"
        "- When the user speaks Persian/Farsi, reply in fluent, natural Farsi with a "
        "native Tehrani accent, pronouncing every Persian sound correctly (خ، غ، ق، ژ, "
        "and the tapped ر) exactly as a native speaker from Tehran would — NOT with an "
        "English accent. In Persian: «فارسی را کاملاً روان و طبیعی صحبت کن، با لهجهٔ "
        "بومیِ تهرانی و تلفّظِ درستِ فارسی، بدون هیچ لهجهٔ خارجی یا انگلیسی.»\n"
        "- Everything you already know about the user and about yourself is "
        "provided ABOVE in this prompt — your identity, the user's profile, your "
        "memories, and today's conversation. Answer questions about the user's "
        "name, your OWN name, and any stored fact or preference DIRECTLY and "
        "instantly from it. NEVER stall or say you need to 'check what we have on "
        "record' for something already provided above.\n"
        "- If the user asks about something genuinely NOT in your provided context, "
        "hand it to the think tool to look it up — do not guess.\n"
        "- You can navigate the user to different pages using the navigate_to tool. "
        "Offer to show them relevant pages when helpful.\n"
        "- You have FULL ACCESS to the user's computer terminal through a connected agent. "
        "You can run shell commands (exec), read files (read_file), write files (write_file), "
        "edit files (edit_file), search files (grep, find, ls), browse the web (web_search, browser), "
        "and more. Use these tools whenever the user asks you to do something on their computer.\n"
        "- When executing terminal commands, briefly tell the user what you're doing.\n"
        "- IMPORTANT: You have a 'think' tool that hands off to your FULL agent — the same brain, "
        "tools, skills, memory, and connected apps (email, calendar, drive, GitHub, and every "
        "connector) you have in text chat. You MUST call it for ANY question, task, action, or "
        "request that needs knowledge, reasoning, research, coding, math, planning, up-to-date facts, "
        "problem-solving, OR an action in the user's tools, accounts, or connected apps. "
        "Only handle simple greetings (hi, hello, bye), yes/no acknowledgments, and casual small talk directly. "
        "For EVERYTHING ELSE, call think(task=<user's full request>). "
        "When you get the result, relay it naturally in your own words as your own work. "
        "NEVER mention the think tool, model switching, reasoning models, or your internal setup to the user.\n"
        "- The user may share their screen with you. When they do, you'll receive periodic "
        "[Screen context: ...] messages describing what's on their screen. Use this visual context "
        "to help them. Don't describe the screen unprompted every time — wait for the user to ask or reference it.\n"
        f"- The current date and time is {now_str}."
    )


def render_onboarding() -> str:
    """The first-conversation script. Relocated verbatim from the relay."""
    return (
        "# ONBOARDING MODE (ACTIVE — THIS IS YOUR FIRST CONVERSATION)\n"
        "You are meeting the user for the very first time. They just deployed you and "
        "you are coming alive! You are centered on their screen, looking at them.\n"
        "IMPORTANT: You do NOT have a name yet. The user will choose your name. "
        "Do NOT introduce yourself with any name — not 'Toup', not 'Agent', nothing.\n\n"

        "## CONVERSATION FLOW (FOLLOW THIS ORDER STRICTLY)\n\n"

        "### Phase 1: Names\n"
        "Your FIRST question MUST be to ask what the user wants to call you and what their name is.\n"
        "Wait for their answer. Then:\n"
        "- Store user's name: memory_store(brain_type='user', category='identity', "
        "content='User name: <name>')\n"
        "- Store your name: memory_store(brain_type='agent', category='agent_soul', "
        "content='My name is <name>')\n\n"

        "### Phase 2: Color\n"
        "After names are set, say something like: \"Now, what color would you like for me? "
        "Pick one from the options on your screen.\"\n"
        "Then IMMEDIATELY call: set_onboarding_phase(phase='color')\n"
        "This will show clickable color circles on the user's screen.\n"
        "WAIT for the user to pick. You will receive a message like "
        "'[COLOR_SELECTED: #hex]'. Acknowledge the color warmly.\n\n"

        "### Phase 3: Deep Profiling\n"
        "Continue naturally, ONE question at a time:\n"
        "- What they primarily need you for — goals, work domain. "
        "Store: brain_type='user', category='goals'\n"
        "- Their preferred language. "
        "Store: brain_type='user', category='preferences'\n"
        "- How they want you to communicate — formal/casual, concise/detailed, "
        "personality preferences. Store: brain_type='agent', category='agent_soul'\n"
        "- Any behavioral rules they want (things to always/never do). "
        "Store: brain_type='agent', category='agent_soul'\n"
        "- Anything else — hobbies, schedule, work style. "
        "Store: brain_type='user', category appropriate\n\n"

        "### Phase 4: Wrap Up\n"
        "After gathering core info (minimum: both names, color, goals, language, "
        "personality preference), summarize what you learned. Then call "
        "finalize_onboarding() to save the complete profiles and finish.\n\n"

        "RULES:\n"
        "- Be warm, enthusiastic, conversational. You're meeting your human!\n"
        "- Ask ONE question at a time. Never dump a list.\n"
        "- Use memory_store for EACH piece of info as you learn it.\n"
        "- Match the user's language automatically.\n"
        "- Do NOT call finalize_onboarding until you have gathered enough info."
    )


async def _load_identities(db: AsyncSession, user_id: str) -> List[Any]:
    from app.db.models import Identity
    from app.db.models.enums import IdentityType

    result = await db.execute(
        select(Identity)
        .where(Identity.user_id == user_id, Identity.is_active.is_(True))
        .order_by(Identity.priority.desc())
    )
    rows = list(result.scalars().all())

    # Duplicate ACTIVE soul rows: a double-write race (two concurrent
    # soul syncs, each finding no existing row) can leave more than one —
    # live on tenant 03cbc72f, two identical "Agent Soul" rows 15ms
    # apart, rendered as TWO "# Core Identity" sections. The legacy
    # builder reads the PLATFORM copy, which its upsert keeps single
    # (save_soul / sync_soul: scalar_one_or_none + update-in-place), so
    # it renders the section once. Keep the row that upsert would own —
    # highest priority, then OLDEST created_at (later upserts update the
    # original row in place; only race clones come after it) — and drop
    # the rest. Read-only: the clones stay in the DB for the operator.
    souls = [r for r in rows if r.identity_type == IdentityType.SOUL.value]
    if len(souls) > 1:
        keep = sorted(
            souls,
            key=lambda r: (
                -(r.priority or 0),
                r.created_at or datetime.max,
                str(r.id or ""),
            ),
        )[0]
        logger.warning(
            "[voice_ctx] %d duplicate active soul rows for %s — rendering "
            "the oldest, a double-write race left the rest",
            len(souls), user_id[:8],
        )
        rows = [
            r for r in rows
            if r.identity_type != IdentityType.SOUL.value or r is keep
        ]
    return rows


async def _resolve_effective_tz(
    db: AsyncSession, user_id: str, tz_override: Optional[str]
) -> Optional[str]:
    """The zone to bucket the day in, or None if we genuinely do not know.

    Returning None matters more than returning a value. `resolve_local_date`
    silently falls back to UTC on an unparseable zone, and
    `resolve_day_chat_id_for_now` consults `User.timezone` only when
    `tz_override` is FALSY — so a client that sends a zone this tzdata does
    not carry gets a UTC day with no further questions asked. In the
    Americas evening that UTC day is already tomorrow, and tomorrow has no
    messages: the session opens having forgotten the whole day.

    So each candidate is VALIDATED before it is trusted, and an
    unresolvable zone is reported as unknown rather than papered over with
    UTC. The caller decides what to do about it.
    """
    import zoneinfo

    for candidate in (tz_override, await _load_user_timezone(db, user_id)):
        if not candidate:
            continue
        try:
            zoneinfo.ZoneInfo(candidate)
            return candidate
        except Exception:
            logger.warning(
                "[voice_ctx] unresolvable timezone %r for %s — not trusting it",
                candidate, user_id[:8],
            )
    return None


async def _load_user_timezone(db: AsyncSession, user_id: str) -> Optional[str]:
    try:
        from app.db.models import User

        user = (await db.execute(
            select(User).where(User.id == user_id)
        )).scalar_one_or_none()
        return getattr(user, "timezone", None) if user else None
    except Exception:
        return None


async def _load_newest_day(db: AsyncSession, user_id: str):
    """(newest DayChat row | None, message rows) — the relay's day feed.

    Mirrors the exact query behind `GET /api/day-chats/{date}/messages`
    for the newest `local_date` (which is what `/api/day-chats?limit=1`
    hands the relay): join for the conversation channel, hide historical
    raw autopilot rows, chronological, capped at DAY_MESSAGES_LIMIT, and
    `channel or "web"` exactly as the endpoint serializes it.
    """
    from app.db.models import Conversation, Message
    from app.db.models.day_chat import DayChat

    newest = (await db.execute(
        select(DayChat)
        .where(DayChat.user_id == user_id)
        .order_by(DayChat.local_date.desc())
        .limit(1)
    )).scalar_one_or_none()
    if newest is None:
        return None, []

    result = await db.execute(
        select(Message.role, Message.content, Conversation.channel)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(
            Message.day_chat_id == newest.id,
            Conversation.channel != "autopilot",
        )
        .order_by(Message.created_at.asc())
        .limit(DAY_MESSAGES_LIMIT)
    )
    rows = [(role, content, channel or "web") for role, content, channel in result.all()]
    return newest, rows


async def _load_agent_name(db: AsyncSession, user_id: str) -> Optional[str]:
    """The tenant's copy of `agent_configs` (SHARED_TABLES).

    Nested so a missing row / missing table cannot poison the caller's
    transaction — the same guard `_build_system_prompt` uses.
    """
    try:
        from app.db.models import AgentConfig

        async with db.begin_nested():
            return (await db.execute(
                select(AgentConfig.agent_name).where(AgentConfig.user_id == user_id)
            )).scalar_one_or_none()
    except Exception:
        return None


async def build_voice_context(
    db: AsyncSession,
    user_id: str,
    *,
    onboarding: bool = False,
    budget_chars: int = 0,
    tz_name: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> VoiceContext:
    """Assemble the Realtime session's instructions from tenant data.

    Args:
        db: the TENANT session. Every read below is the same row text
            chat reads — that is the point of the move.
        user_id: tenant owner.
        onboarding: append the first-conversation script.
        budget_chars: total character budget for the trimmable blocks
            (0 = no trimming, which is what the relay passes when
            VOICE_REALTIME_V2 is off).
        tz_name: IANA zone. None falls back to `User.timezone`, exactly
            as `resolve_day_chat_id_for_now` does for every other caller.
        now_utc: freeze the instant. Defaults to the real clock.

    Returns a `VoiceContext`; it never raises. A leg that FAILED is named
    in `degraded`; a leg that succeeded with nothing to show is named in
    `empty`. Conflating those two is how a real outage hides behind an
    alarm that fires every morning.
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    sections: Dict[str, str] = {}
    degraded: List[str] = []
    empty: List[str] = []

    # ── 1. Persona ────────────────────────────────────────────────────
    identity_failed = False
    try:
        identities = await _load_identities(db, user_id)
    except Exception as exc:
        logger.warning("[voice_ctx] identity load failed for %s: %s", user_id[:8], exc)
        identities = []
        identity_failed = True
    if identity_failed:
        degraded.append("identity")
    elif not identities:
        empty.append("identity")

    identity_text, has_soul = render_identity_sections(identities)
    sections["identity"] = identity_text
    if not has_soul:
        logger.warning("[voice_ctx] no soul document for %s — default persona", user_id[:8])

    # Render exactly what the TENANT row says, including nothing.
    #
    # An earlier version of this file defaulted an empty tenant name to
    # "Agent" to match what legacy renders, on the premise that the
    # platform holds "Agent" for every never-renamed agent. Measured
    # against production, that premise is false for 5 of the 45 bound
    # tenants, whose PLATFORM agent_name is itself NULL — they never
    # saved the Soul page at all. Four of those five already agree with
    # legacy today (both sides render the nameless anchor), and the
    # default would have broken all four AND made their agents announce
    # "Your name is Agent" after the flip, to a user who never named one.
    #
    # The remaining divergence — platform "Agent", tenant NULL, on three
    # tenants — is left INTENDED rather than matched. Legacy tells those
    # users their agent is called "Agent"; the flip stops it. Matching
    # legacy there would mean reproducing a stub name in voice, which is
    # the same outcome the agent_name backfill was explicitly forbidden
    # from producing in data.
    sections["identity_anchor"] = render_identity_anchor(
        await _load_agent_name(db, user_id), fmt="voice"
    )

    # ── 2. Memory — voice's dump shape, tenant rows ───────────────────
    agent_mems: List[Any] = []
    user_mems: List[Any] = []
    try:
        from app.services.memory_service import MemoryService

        svc = MemoryService(db)
        agent_mems, _ = await svc.list_memories(
            user_id=user_id, limit=VOICE_MEMORIES_LIMIT, brain_type="agent",
        )
        user_mems, _ = await svc.list_memories(
            user_id=user_id, limit=VOICE_MEMORIES_LIMIT, brain_type="user",
        )
        memories_failed = False
    except Exception as exc:
        logger.warning("[voice_ctx] memory load failed for %s: %s", user_id[:8], exc)
        memories_failed = True

    if agent_mems:
        text = _render_brain("# Agent Brain (Permanent Knowledge)", agent_mems)
        if budget_chars:
            text = cap_chars(text, int(budget_chars * BUDGET_SHARE_AGENT_BRAIN), keep="head")
        sections["agent_brain"] = text
    if user_mems:
        # The referent matters more in voice than anywhere: entries are
        # written in the user's second person, and without this parenthesis
        # "- [identity] Your name is Nariman" reads as the MODEL's name.
        text = _render_brain(
            "# User Brain (what you know about the user — entries speak TO "
            "the user: 'you'/'your' in them means the USER, not you)",
            user_mems,
        )
        if budget_chars:
            text = cap_chars(text, int(budget_chars * BUDGET_SHARE_USER_BRAIN), keep="head")
        sections["user_brain"] = text
    if memories_failed:
        degraded.append("memories")
    elif not agent_mems and not user_mems:
        empty.append("memories")
    elif not agent_mems or not user_mems:
        # Granular signal for the operator; `degraded` stays coarse so a
        # consumer does not have to know the brain taxonomy.
        logger.info(
            "[voice_ctx] one brain empty for %s (agent=%d user=%d)",
            user_id[:8], len(agent_mems), len(user_mems),
        )

    # ── 3. Today — the relay's day feed, reproduced byte for byte ─────
    # Newest existing day chat, the date-guard header, the relay's line
    # format (see the module docstring's D4 note: the flip criterion is
    # fingerprint equality, so this leg mirrors the shipped relay rather
    # than improving on it). The zone decides only the LABELLING: with a
    # resolvable zone and a transcript that is not from the user's local
    # today, the header names its real date; with no zone it makes no
    # date claim. #488 — a previous day narrated as today — stays fixed
    # in both cases.
    day_date: Optional[str] = None
    day_text = ""
    tz_effective = await _resolve_effective_tz(db, user_id, tz_name)
    if not tz_effective:
        # Operator signal only — the prompt itself simply carries no
        # date-labelled header. Not `degraded="day"`: the transcript IS
        # served.
        degraded.append("day_timezone")
        logger.warning(
            "[voice_ctx] no resolvable timezone for %s — day block kept, "
            "date labelling disabled", user_id[:8],
        )
    try:
        from zoneinfo import ZoneInfo

        from app.config import settings as _settings

        newest, rows = await _load_newest_day(db, user_id)
        if newest is not None and newest.local_date is not None:
            day_date = newest.local_date.isoformat()
        local_today = None
        if tz_effective and getattr(_settings, "voice_day_context_date_guard", True):
            local_today = now_utc.astimezone(ZoneInfo(tz_effective)).strftime("%Y-%m-%d")
        day_text = render_day_history(rows, day_date, local_today)
    except Exception as exc:
        # A raise here under the relay meant a Realtime session opened
        # with NO instructions at all — the exact 2026-07-31 shape
        # `degraded` exists to prevent.
        logger.warning("[voice_ctx] day leg failed for %s: %s", user_id[:8], exc)
        degraded.append("day")

    if day_text:
        if budget_chars:
            day_text = cap_chars(day_text, int(budget_chars * BUDGET_SHARE_DAY_HISTORY), keep="tail")
        sections["day_history"] = day_text
    elif "day" not in degraded:
        # Succeeded and there is simply nothing yet — the first voice call
        # of the morning. Not an alarm.
        empty.append("day")

    # ── 4. Channel document + onboarding ──────────────────────────────
    sections["voice_mode"] = render_voice_mode(now_utc)
    if onboarding:
        sections["onboarding"] = render_onboarding()

    if degraded:
        logger.warning(
            "[voice_ctx] voice context DEGRADED — no %s for user %s; the model "
            "is answering without it", ", ".join(degraded), user_id[:8],
        )

    instructions = "\n\n".join(
        sections[k] for k in VOICE_SECTION_ORDER if sections.get(k)
    )
    return VoiceContext(
        instructions=instructions,
        day_date=day_date,
        sections=sections,
        degraded=degraded,
        empty=empty,
    )
