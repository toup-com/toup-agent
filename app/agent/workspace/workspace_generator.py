"""
Workspace Generator — Auto-generate personalized workflows after onboarding.

Reads the user's goals from their memories, matches them to pre-built template
packs using the strongest available LLM, and creates workflows in the DB.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# ── Template pack cache ──────────────────────────────────────────────
_packs_cache: Optional[dict] = None
_TEMPLATES_DIR = Path(__file__).parent / "templates"


def load_template_packs() -> dict:
    """Load all JSON template packs from the templates/ directory."""
    global _packs_cache
    if _packs_cache is not None:
        return _packs_cache

    packs = {}
    if not _TEMPLATES_DIR.is_dir():
        logger.warning("[WORKSPACE] Templates directory not found: %s", _TEMPLATES_DIR)
        return packs

    for f in sorted(_TEMPLATES_DIR.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            name = data.get("name", f.stem)
            packs[name] = data
        except Exception as e:
            logger.warning("[WORKSPACE] Failed to load template pack %s: %s", f.name, e)

    _packs_cache = packs
    logger.info("[WORKSPACE] Loaded %d template packs: %s", len(packs), list(packs.keys()))
    return packs


# ── Model selection ──────────────────────────────────────────────────

def get_strongest_model() -> tuple[str, str]:
    """Return (provider, model) — the strongest model the user has access to."""
    if getattr(settings, "anthropic_api_key", None):
        return ("anthropic", "claude-opus-4-6")
    if getattr(settings, "openai_api_key", None):
        model = getattr(settings, "agent_model", "gpt-4o")
        return ("openai", model)
    return ("openai", "gpt-4o-mini")


# ── LLM matching ────────────────────────────────────────────────────

async def _llm_match_openai(user_goals: str, pack_descriptions: str, model: str) -> list[str]:
    """Match goals to packs using OpenAI."""
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You match a user's description of what they need an AI agent for "
                    "to pre-built workspace template packs. Return ONLY a JSON array of "
                    "1-3 pack names that best match. Example: [\"freelancer\", \"developer\"]\n\n"
                    f"Available packs:\n{pack_descriptions}"
                ),
            },
            {"role": "user", "content": user_goals},
        ],
        temperature=0,
        max_tokens=200,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content or "[]"
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("packs", data.get("pack_names", []))
    except json.JSONDecodeError:
        pass
    return []


async def _llm_match_anthropic(user_goals: str, pack_descriptions: str, model: str) -> list[str]:
    """Match goals to packs using Anthropic."""
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    resp = await client.messages.create(
        model=model,
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": (
                    "Match this user's description to 1-3 workspace template packs.\n\n"
                    f"Available packs:\n{pack_descriptions}\n\n"
                    f"User's goals: {user_goals}\n\n"
                    "Return ONLY a JSON array of pack names. Example: [\"freelancer\", \"developer\"]"
                ),
            },
        ],
    )

    text = resp.content[0].text if resp.content else "[]"
    try:
        # Extract JSON from response (may have markdown wrapping)
        if "[" in text:
            start = text.index("[")
            end = text.rindex("]") + 1
            return json.loads(text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _keyword_match(user_goals: str, packs: dict) -> list[str]:
    """Fallback: simple keyword matching when LLM is unavailable."""
    goals_lower = user_goals.lower()
    scores = {}
    for name, pack in packs.items():
        score = sum(1 for kw in pack.get("keywords", []) if kw in goals_lower)
        if score > 0:
            scores[name] = score

    if not scores:
        return ["personal_productivity"]

    return sorted(scores, key=scores.get, reverse=True)[:3]


async def match_packs(user_goals: str) -> list[str]:
    """Match user goals to 1-3 template packs using the strongest available model."""
    packs = load_template_packs()
    if not packs:
        return []

    pack_descriptions = "\n".join(
        f"- {name}: {p.get('description', '')}" for name, p in packs.items()
    )

    provider, model = get_strongest_model()
    logger.info("[WORKSPACE] Matching goals with %s/%s", provider, model)

    try:
        if provider == "anthropic":
            matched = await _llm_match_anthropic(user_goals, pack_descriptions, model)
        else:
            matched = await _llm_match_openai(user_goals, pack_descriptions, model)

        # Validate pack names exist
        valid = [m for m in matched if m in packs]
        if valid:
            logger.info("[WORKSPACE] LLM matched packs: %s", valid)
            return valid
        logger.warning("[WORKSPACE] LLM returned no valid packs, falling back to keywords")
    except Exception as e:
        logger.warning("[WORKSPACE] LLM matching failed (%s), falling back to keywords", e)

    return _keyword_match(user_goals, packs)


# ── Workspace generation ─────────────────────────────────────────────

async def generate_workspace(user_id: str):
    """
    Main entry point: read user goals, match to template packs, create workflows.

    Called as a background task after onboarding completes.
    """
    try:
        from app.db.database import async_session_maker
        from app.db.models import Workflow, Memory

        async with async_session_maker() as db:
            from sqlalchemy import select

            # Guard: skip if workflows already exist
            existing = (await db.execute(
                select(Workflow.id).where(Workflow.user_id == user_id).limit(1)
            )).scalar_one_or_none()
            if existing:
                logger.info("[WORKSPACE] Workflows already exist for %s, skipping", user_id[:8])
                return

            # Read user goals from memories
            goals_rows = (await db.execute(
                select(Memory.content).where(
                    Memory.user_id == user_id,
                    Memory.brain_type == "user",
                    Memory.category == "goals",
                )
            )).scalars().all()

            if not goals_rows:
                # Also check for general user memories that mention goals
                goals_rows = (await db.execute(
                    select(Memory.content).where(
                        Memory.user_id == user_id,
                        Memory.brain_type == "user",
                    ).limit(20)
                )).scalars().all()

            if not goals_rows:
                logger.info("[WORKSPACE] No goals found for %s, using default pack", user_id[:8])
                matched_packs = ["personal_productivity"]
            else:
                goals_text = "\n".join(goals_rows)
                matched_packs = await match_packs(goals_text)

            # Load template packs and create workflows
            all_packs = load_template_packs()
            created = 0

            for pack_name in matched_packs:
                pack = all_packs.get(pack_name)
                if not pack:
                    continue

                for wf_template in pack.get("workflows", []):
                    workflow = Workflow(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        name=wf_template["name"],
                        description=wf_template.get("description", ""),
                        status="draft",
                        nodes_json=json.dumps(wf_template["nodes_json"]),
                        edges_json=json.dumps(wf_template["edges_json"]),
                        run_count=0,
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    db.add(workflow)
                    created += 1

            await db.commit()
            logger.info(
                "[WORKSPACE] Generated %d workflows from %d packs for %s: %s",
                created, len(matched_packs), user_id[:8], matched_packs,
            )

    except Exception as e:
        logger.error("[WORKSPACE] Failed to generate workspace for %s: %s", user_id[:8], e)
