"""Session-thread conversation intelligence (Round 29).

An automation's session thread is a chat like any other — the R28
runner honors the address instead of forking. What was missing is the
agent KNOWING it is standing inside an automation when it answers
there. This module supplies both halves:

  - `build_automation_context` + `prompt_section` — the system-prompt
    section for a turn on an `automation`-channel conversation: what
    this automation is (name, standing rule, status), what the fact
    ledger already knows, and the interview posture — ask for the
    context the automation is missing, ONE question at a time, and
    never promise a send (drafts only; the rail is engine-enforced).

  - `extract_and_record_facts` — the post-turn writer. Automation
    session turns run THIS instead of the global curator
    (CONTRACTS-R29 §4: the write seam `facts.record`
    projects to the brain itself; running `curate_turn` beside it
    would double-file the same facts). One small pinned-model JSON
    call (`settings.memory_extraction_model` — never `model=None` on
    a background path), then one `record` per category, then the
    "Memory updated · N facts" chip.

Raw tool names never appear in anything this module composes — the
prompt section describes the rule via the automation's own
`rule_text`/description vocabulary, with the verbs module supplying
the sentence when it can.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Categories the extractor may file under, in ledger order. Domain
#: slugs are added per automation at call time.
CANONICAL_CATEGORIES = ("people", "preferences", "deadlines")

_MAX_FACTS_PER_TURN = 5
_FACT_MAX_LEN = 300


async def build_automation_context(db, conversation) -> Optional[dict]:
    """The automation behind this conversation, shaped for the prompt
    section — or None when the conversation isn't an automation session
    (the caller's common case; one metadata parse, no query)."""
    from .session import SESSION_CHANNEL, automation_id_of

    if getattr(conversation, "channel", None) != SESSION_CHANNEL:
        return None
    automation_id = automation_id_of(conversation)
    if not automation_id:
        return None
    try:
        from app.db.models import Automation

        automation = await db.get(Automation, automation_id)
        if automation is None:
            return None
        rule_text = _rule_text(automation)
        facts = await _load_facts_grouped(db, automation_id)
        return {
            "automation_id": automation_id,
            "name": automation.name,
            "rule_text": rule_text,
            "status": automation.status,
            "paused_reason": automation.paused_reason,
            "domain": getattr(automation, "domain", None),
            "last_error": automation.last_error,
            "facts": facts,
        }
    except Exception as e:  # noqa: BLE001 — context is a companion
        logger.warning(
            "[automations] session context load failed conv=%s: %s",
            str(getattr(conversation, "id", "?"))[:8], e,
        )
        return None


def _rule_text(automation) -> str:
    """One plain sentence for what this automation does — the verbs
    module when available, the stored description otherwise, never the
    spec JSON."""
    try:
        from app.services.automation_verbs import rule_sentence

        spec_raw = automation.spec_json
        if isinstance(spec_raw, str):
            try:
                spec_raw = json.loads(spec_raw)
            except (ValueError, TypeError):
                spec_raw = None
        if isinstance(spec_raw, dict):
            sentence = rule_sentence(spec_raw)
            if sentence:
                return str(sentence)
    except Exception:  # noqa: BLE001 — fall through to the description
        pass
    return (automation.description or automation.name or "").strip()


async def _load_facts_grouped(db, automation_id: str) -> dict[str, list[str]]:
    """Ledger facts grouped by category, canonical order first. Empty
    dict when the table predates the R29-A half."""
    grouped: dict[str, list[str]] = {}
    try:
        from sqlalchemy import select
        from app.db.models import AutomationFact

        rows = (await db.execute(
            select(AutomationFact)
            .where(AutomationFact.automation_id == automation_id)
            .order_by(AutomationFact.created_at.asc())
        )).scalars().all()
        for row in rows:
            text = (row.text or "").strip()
            if text:
                grouped.setdefault(row.category, []).append(text)
    except ImportError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.warning("[automations] facts load failed: %s", e)
    ordered: dict[str, list[str]] = {}
    for cat in CANONICAL_CATEGORIES:
        if cat in grouped:
            ordered[cat] = grouped.pop(cat)
    ordered.update(dict(sorted(grouped.items())))
    return ordered


def _friendly_error(raw: object) -> str:
    """One plain clause for the last failure — never the raw engine
    string (it has carried tool ids and provider text into this prompt
    before; D-19's sibling leak). Known classes get their §5.8 phrasing;
    anything else the honest generic."""
    text = str(raw or "").lower()
    if not text:
        return "the last run could not finish"
    if "expired" in text or "token" in text or "401" in text or "reauth" in text:
        return "the access it had ran out"
    if "forbidden" in text:
        return "it tried something automations never do, and was stopped"
    if "timeout" in text or "cap" in text:
        return "the last run took too long and was stopped"
    return "the last run could not finish"


def prompt_section(ctx: dict) -> str:
    """The `automation_session` system-prompt section — the R30 thread
    posture: grounded answers from the run record, one status claim per
    reply, memory first, the interview kept from R29."""
    lines = [
        "## This conversation is an automation's thread",
        f'The user is inside the thread of their automation '
        f'"{ctx["name"]}". This thread holds every run plus their '
        "questions; the main chat only ever receives this automation's "
        "notification card — never repeat a run's findings there.",
    ]
    if ctx.get("rule_text"):
        lines.append(f"Its standing rule: {ctx['rule_text']}")

    status = ctx.get("status")
    if status == "error":
        lines.append(
            "Its status, stated once: PAUSED after repeated failures — "
            f"{_friendly_error(ctx.get('last_error'))}. Help the user "
            "fix the cause, then re-arm it."
        )
    elif status == "paused":
        lines.append(
            "Its status, stated once: paused by the user. It keeps its "
            "setup and memory and will not run until resumed."
        )
    lines.append(
        "State its status from the line above, ONCE per reply, and "
        "never two ways — a reply that says both 'active' and 'paused' "
        "is a defect."
    )

    facts = ctx.get("facts") or {}
    if any(facts.values()):
        lines.append("\nWhat its memory already knows:")
        for cat, texts in facts.items():
            for text in texts[:8]:
                lines.append(f"- [{cat}] {text}")
    else:
        lines.append(
            "\nIts memory is EMPTY — it knows nothing about the people, "
            "preferences, or deadlines it should be tuned to."
        )

    lines.append(
        "\nHow to behave here:\n"
        "- \"Ask why it did that\" is answered from THIS thread's run "
        "record: cite the item by name and give the reason that was "
        "recorded with it. If the record does not contain the answer, "
        "say so plainly — never invent one, never quietly re-run. If "
        "answering needs new reading, say what you will look at and do "
        "it here in this thread, never in the main chat.\n"
        "- Before answering anything about a person, a channel, a "
        "ticket or a past run, check memory first (memory recall / "
        "memory_search) — the platform memory holds everything the "
        "automations learned and did.\n"
        "- Interview for the context the automation is missing: who "
        "matters (people), how the user wants things handled "
        "(preferences), and dates that matter (deadlines). Ask ONE "
        "question at a time, conversationally — never a form.\n"
        "- Anything durable the user tells you here is saved to this "
        "automation's memory automatically after your reply; don't "
        "narrate the saving, and never re-ask what the memory above "
        "already answers. What an automation IS — its schedule, its "
        "status, its rule — is never a memory.\n"
        "- Be honest about actions: this automation can stage drafts "
        "and post summaries, but it NEVER sends mail — never promise "
        "otherwise. If a run is waiting on the user's approval, "
        "approving it is their tap, not your call.\n"
        "- Describe what runs did in plain words; never expose "
        "internal tool or step identifiers, and never use engine "
        "jargon for it (no 'Mission Control', no 'polling', no "
        "percent-complete talk — it reads, it drafts, it tells you)."
    )
    return "\n".join(lines)


def _extraction_prompt(ctx: dict, user_text: str, assistant_text: str) -> str:
    existing = json.dumps(ctx.get("facts") or {}, ensure_ascii=False)
    return (
        "You file durable facts about the user into one platform "
        f'memory. The exchange happened inside their automation '
        f'"{ctx["name"]}" (its rule: {ctx.get("rule_text") or "—"}).\n'
        f"Already known: {existing}\n\n"
        "From the exchange below, extract facts WORTH KEEPING. For "
        "each:\n"
        '- "category": people (who matters and how) · team_workspace '
        "(channels, ownership, team habits) · your_time (blocks, "
        "holds, when things reach the user) · work_you_own (surfaces, "
        "tickets, priorities) · noise_filters (what never surfaces).\n"
        '- "scope": "automation" when it only matters to this '
        'automation\'s work; "global" when it is about the person.\n'
        '- "subject": the person/channel/ticket/repo it is about, or '
        "null.\n"
        '- "why": the evidence in one second-person sentence '
        '("You said Sarah is your boss.").\n'
        "Rules: only what the USER stated or clearly confirmed this "
        "turn; nothing inferred; nothing already known (either scope). "
        "NEVER file what an automation is or does, its schedule, its "
        "status, or run outcomes. One short self-contained sentence "
        f"each; dates absolute; at most {_MAX_FACTS_PER_TURN}; an "
        "empty list is the right answer for small talk.\n\n"
        f"USER: {user_text[:2000]}\n\nASSISTANT: {assistant_text[:1500]}\n\n"
        'Reply as JSON: {"facts": [{"text", "category", "scope", '
        '"subject", "why"}]}'
    )


async def extract_and_record_facts(
    db,
    *,
    user_id: str,
    ctx: dict,
    user_text: str,
    assistant_text: str,
) -> int:
    """The post-turn writer for automation session turns. Returns the
    number of facts saved (0 on any failure — this is a background
    companion, never a veto)."""
    text = (user_text or "").strip()
    if not text:
        return 0
    automation_id = ctx["automation_id"]
    try:
        from app.config import settings
        from app.services.llm_service import get_llm_service

        response = await get_llm_service().complete_with_json(
            messages=[{
                "role": "user",
                "content": _extraction_prompt(ctx, text, assistant_text or ""),
            }],
            model=settings.memory_extraction_model,
            temperature=0.0,
        )
        parsed: Any = response
        if hasattr(response, "content"):
            raw = (response.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
            parsed = json.loads(raw)
        items = parsed.get("facts") if isinstance(parsed, dict) else None
        if not isinstance(items, list):
            return 0

        from .curator_v2 import file_facts, normalize_candidate

        # normalize_candidate applies the ND-2/ND-3 refusal gate: a
        # definition or a run-status sentence is refused here, not just
        # dropped by A's migration.
        candidates = [
            fact for fact in (
                normalize_candidate(item)
                for item in items[:_MAX_FACTS_PER_TURN]
            ) if fact is not None
        ]
        if not candidates:
            return 0
        saved = await file_facts(
            db,
            user_id=user_id,
            facts=candidates,
            automation_id=automation_id,
            domain=ctx.get("domain"),
            source="agent",
        )
        if saved:
            from .session import emit_memory_update

            await emit_memory_update(
                db,
                user_id=user_id,
                automation_id=automation_id,
                count=saved,
                title=ctx.get("name"),
            )
        return saved
    except ImportError:
        # automation_facts ships with the R29-A half; until the rebase
        # the interview still talks, it just cannot file.
        logger.info("[automations] interview extractor: facts seam "
                    "unavailable; nothing recorded")
        return 0
    except Exception as e:  # noqa: BLE001 — background companion
        logger.warning(
            "[automations] interview extraction failed automation=%s: "
            "%s: %s",
            str(automation_id)[:8], type(e).__name__, str(e)[:200],
        )
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        return 0
