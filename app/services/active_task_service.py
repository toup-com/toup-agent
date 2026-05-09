"""
Active Task Service — extract, decay, and retrieve ongoing work items.

Active tasks are memories with category="active_task" that represent what
the user is currently working on: "debugging the CSS sidebar", "building
a React component", "waiting for the deploy to finish".

Key behaviors:
  - Extracted from conversations by pattern matching + LLM extraction
  - 7-day TTL from creation or last reinforcement
  - Decayed (archived) if not reinforced within TTL
  - Always injected into system prompt regardless of intent classification
  - Placed in a dedicated <active_tasks> block, separate from general memories
"""

import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from sqlalchemy import select, and_, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# TTL: 7 days from creation or last reinforcement
ACTIVE_TASK_TTL_DAYS = 7

# Patterns that signal an active task in user messages
_ACTIVE_TASK_PATTERNS = [
    re.compile(r"(?:i'm|i am|we're|we are)\s+(?:working on|building|debugging|fixing|implementing|writing|creating|designing|testing|deploying|setting up|configuring|migrating|refactoring|investigating|researching)", re.IGNORECASE),
    re.compile(r"(?:still|currently)\s+(?:need to|have to|trying to|waiting for|stuck on|dealing with)", re.IGNORECASE),
    re.compile(r"(?:the|my|our)\s+(?:bug|issue|problem|task|project|feature|fix)\s+(?:is|was|involves)", re.IGNORECASE),
    re.compile(r"(?:next step|next thing|todo|to-do|to do)\s+(?:is|would be|should be)", re.IGNORECASE),
    re.compile(r"(?:i need to|i want to|i have to|i should)\s+(?:finish|complete|fix|build|deploy|test|review|check|update)", re.IGNORECASE),
    re.compile(r"(?:let's|let me|can you help me)\s+(?:continue|keep going|finish|work on|debug|fix)", re.IGNORECASE),
    re.compile(r"(?:in the middle of|halfway through|almost done with|started working on)", re.IGNORECASE),
]


def detect_active_tasks(user_message: str, assistant_response: str) -> List[str]:
    """Extract active task descriptions from a conversation turn using patterns.

    Returns a list of task description strings found in the user message.
    This is a fast, non-LLM extraction for the most common patterns.
    """
    tasks = []
    for pattern in _ACTIVE_TASK_PATTERNS:
        match = pattern.search(user_message)
        if match:
            # Extract a reasonable snippet around the match
            start = max(0, match.start() - 10)
            end = min(len(user_message), match.end() + 100)
            snippet = user_message[start:end].strip()
            # Clean up: take until end of sentence
            for delimiter in ['.', '!', '?', '\n']:
                idx = snippet.find(delimiter, match.end() - start)
                if idx > 0:
                    snippet = snippet[:idx + 1]
                    break
            if len(snippet) > 15:  # Skip very short fragments
                tasks.append(snippet)

    return tasks[:3]  # Max 3 active tasks per turn


async def store_active_task(
    db: AsyncSession,
    user_id: str,
    content: str,
    source_message_id: Optional[str] = None,
) -> Optional[str]:
    """Store or reinforce an active_task memory.

    If a similar active task already exists (fuzzy match on content),
    reinforce it (reset TTL). Otherwise create a new one.

    Returns the memory ID.
    """
    from app.db.models.memory import Memory

    # Check for existing similar active tasks (simple word overlap)
    existing = (await db.execute(
        select(Memory).where(
            and_(
                Memory.user_id == user_id,
                Memory.category == "active_task",
                Memory.is_active == True,
                Memory.is_deleted == False,
            )
        )
    )).scalars().all()

    content_words = set(content.lower().split())

    for mem in existing:
        mem_words = set(mem.content.lower().split())
        overlap = len(content_words & mem_words)
        union = len(content_words | mem_words)
        if union > 0 and overlap / union > 0.5:  # >50% word overlap = same task
            # Reinforce — reset TTL clock
            mem.last_reinforced_at = datetime.utcnow()
            mem.strength = 1.0  # Reset strength
            mem.content = content  # Update to latest phrasing
            mem.updated_at = datetime.utcnow()
            await db.flush()
            logger.info("[active_task] Reinforced: %s", content[:60])
            return mem.id

    # Create new active task memory
    mem_id = str(uuid.uuid4())
    mem = Memory(
        id=mem_id,
        user_id=user_id,
        brain_type="user",
        content=content,
        category="active_task",
        memory_type="semantic",
        importance=0.9,  # High — always injected
        confidence=1.0,
        strength=1.0,
        memory_level="working",
        decay_rate=0.05,  # Slow decay — preserved for 7 days
        source_message_id=source_message_id,
        source_type="active_task_extraction",
        is_active=True,
    )
    db.add(mem)
    await db.flush()
    logger.info("[active_task] Created: %s", content[:60])
    return mem_id


async def decay_expired_tasks(db: AsyncSession, user_id: str) -> int:
    """Archive active_task memories that haven't been reinforced within TTL.

    Returns count of archived tasks.
    """
    from app.db.models.memory import Memory

    cutoff = datetime.utcnow() - timedelta(days=ACTIVE_TASK_TTL_DAYS)

    result = await db.execute(
        select(Memory).where(
            and_(
                Memory.user_id == user_id,
                Memory.category == "active_task",
                Memory.is_active == True,
                Memory.is_deleted == False,
            )
        )
    )
    tasks = result.scalars().all()

    archived = 0
    for task in tasks:
        # Use last_reinforced_at if available, otherwise created_at
        last_active = task.last_reinforced_at or task.created_at
        if last_active and last_active < cutoff:
            task.is_active = False
            task.strength = 0.0
            archived += 1
            logger.info("[active_task] Archived (TTL expired): %s", task.content[:60])

    if archived:
        await db.flush()

    return archived


async def get_active_tasks(db: AsyncSession, user_id: str) -> List[Dict]:
    """Retrieve all active (non-expired, non-archived) task memories.

    These are always injected into the system prompt regardless of
    intent classification or hybrid search ranking.
    """
    from app.db.models.memory import Memory

    result = await db.execute(
        select(Memory).where(
            and_(
                Memory.user_id == user_id,
                Memory.category == "active_task",
                Memory.is_active == True,
                Memory.is_deleted == False,
            )
        ).order_by(Memory.updated_at.desc())
    )
    tasks = result.scalars().all()

    return [
        {
            "id": t.id,
            "content": t.content,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "last_reinforced_at": t.last_reinforced_at.isoformat() if t.last_reinforced_at else None,
            "strength": t.strength,
        }
        for t in tasks
    ]


def build_active_tasks_block(tasks: List[Dict]) -> str:
    """Build the <active_tasks> system prompt section.

    Always injected when tasks exist — regardless of intent classification.

    F4 (2026-05-08): the previous instruction ("Reference them when relevant")
    was too passive. The model deferred to "is it relevant?" and almost
    never surfaced an open thread on its own. The new instruction primes
    the proactive-recall behaviour the founder explicitly asked for: a
    friend who picks up a thread without being asked to.

    Important constraint: the model MUST NOT make this feel scripted.
    Ask, don't list. At most one mention per session. Pair with the
    voice_rules anti-chatbot guards higher in the prompt — same posture.
    """
    if not tasks:
        return ""

    lines = [
        "\n<active_tasks>",
        "Open threads with this user — things they were working on, "
        "stuck on, or said they'd come back to. These persist across "
        "days and channels. Don't list them; that's not what a friend "
        "does. Instead:",
        "",
        "- **First message of the day** (or a long gap since last turn): "
        "if one of these threads still feels unresolved, ask about it "
        "naturally as part of your reply. \"Did the Stripe webhook thing "
        "ever sort itself out?\" — that kind of beat. ONE thread max, the "
        "most recently reinforced. Don't bring it up if their opener is "
        "clearly about something else important.",
        "",
        "- **Mid-conversation:** if their current message connects to one "
        "of these threads (even loosely), reference it directly. \"Right, "
        "that ties back to the migration you were stuck on earlier.\"",
        "",
        "- **Never say** \"I see you're working on...\" or \"I remember "
        "that you...\" — those are robot phrases. Just ask, or just say "
        "the thing, the way a friend would.",
        "",
        "- **If a thread is done**, drop it: don't keep asking about a "
        "task they've moved past. When in doubt, the recently-reinforced "
        "ones are the live ones.",
        "",
        "Open threads (most-recently-reinforced first):",
    ]
    for t in tasks[:10]:  # Max 10 active tasks
        age = ""
        if t.get("last_reinforced_at"):
            age = f" (last mentioned: {t['last_reinforced_at'][:10]})"
        elif t.get("created_at"):
            age = f" (since: {t['created_at'][:10]})"
        lines.append(f"- {t['content']}{age}")
    lines.append("</active_tasks>")

    return "\n".join(lines) + "\n"
