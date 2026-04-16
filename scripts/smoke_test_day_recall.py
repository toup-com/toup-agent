#!/usr/bin/env python3
"""
Smoke test for the day-recall feature against staging.

Exercises the screenshot scenario end-to-end:

  1. Seeds a fake "yesterday" DayChat for the target user across web + telegram,
     including an identifiable lesson thread.
  2. Forces an archival summary generation via the same code path the hourly
     job uses.
  3. Starts a new day turn.
  4. Sends "do you remember what we talked about yesterday?" and confirms the
     agent's response (a) references recalled content, and (b) is not the
     "I don't have a clear memory" failure mode.
  5. Sends "make me a quiz from what you taught me yesterday about <topic>"
     and confirms the quiz is grounded in the seeded content (topic keyword
     appears in the quiz).

Usage:
  STAGING_BASE_URL=https://staging.toup.ai \\
  STAGING_USER_EMAIL=ceo@ete.bar \\
  STAGING_USER_PASSWORD=... \\
  TARGET_USER_ID=<uuid-of-test-user> \\
    python scripts/smoke_test_day_recall.py

The test user must be a real account on the staging environment with
ENABLE_DAY_RECALL=true. Exits 0 on pass, non-zero on any failure.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone


# ── Config ────────────────────────────────────────────────────────────

TOPIC = "chain rule"  # The thing "taught yesterday" that the quiz must cover.
LESSON = [
    ("user", "teach me the chain rule"),
    ("assistant",
     "The chain rule states that d/dx[f(g(x))] = f'(g(x)) · g'(x). "
     "Example: if y = sin(x^2), then dy/dx = cos(x^2) · 2x."),
    ("user", "give me a harder example"),
    ("assistant",
     "Let y = e^(3x^2 + 1). Then dy/dx = e^(3x^2 + 1) · 6x. "
     "Outer derivative of e^u is e^u; inner derivative of (3x^2 + 1) is 6x."),
    ("user", "what if it's nested three deep?"),
    ("assistant",
     "Apply the chain rule one layer at a time. For y = sin(cos(x^2)): "
     "dy/dx = cos(cos(x^2)) · (-sin(x^2)) · 2x."),
]
TELEGRAM_NOISE = [
    ("user", "btw tell me a joke later"),
    ("assistant", "noted"),
]


# ── DB seeding (direct, same path the real agent uses) ───────────────

async def seed_yesterday(user_id: str, user_tz: str = "UTC") -> str:
    """Seed a DayChat + sessions + messages for yesterday in the user's tz.

    Returns the DayChat id.
    """
    # Import lazily so non-DB commands (--dry-run) don't pull settings.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.db.database import async_session_maker
    from app.db.models.day_chat import DayChat
    from app.db.models import Conversation, Message
    from sqlalchemy import text, select
    import zoneinfo

    tz = zoneinfo.ZoneInfo(user_tz) if user_tz else timezone.utc
    yesterday_local = (datetime.now(tz) - timedelta(days=1)).date()
    base_ts = datetime(
        yesterday_local.year, yesterday_local.month, yesterday_local.day,
        10, 0, 0, tzinfo=timezone.utc,
    )

    dc_id = str(uuid.uuid4())
    web_sess_id = str(uuid.uuid4())
    tg_sess_id = str(uuid.uuid4())

    async with async_session_maker() as db:
        # Idempotent: delete any previous seeds for (user, yesterday) so reruns
        # don't pile up duplicate conversations.
        await db.execute(text(
            "DELETE FROM messages WHERE day_chat_id IN "
            "(SELECT id FROM day_chats WHERE user_id = :uid AND local_date = :d)"
        ), {"uid": user_id, "d": yesterday_local})
        await db.execute(text(
            "DELETE FROM conversations WHERE day_chat_id IN "
            "(SELECT id FROM day_chats WHERE user_id = :uid AND local_date = :d)"
        ), {"uid": user_id, "d": yesterday_local})
        await db.execute(text(
            "DELETE FROM day_chats WHERE user_id = :uid AND local_date = :d"
        ), {"uid": user_id, "d": yesterday_local})
        await db.commit()

        # Fresh DayChat.
        db.add(DayChat(
            id=dc_id, user_id=user_id, local_date=yesterday_local,
            timezone=user_tz, started_at=base_ts, last_message_at=base_ts,
            message_count=0, total_tokens=0,
            summary_status="up_to_date",
            archival_summary_status="not_needed",
        ))
        # Two conversations: web (lesson) and telegram (noise).
        db.add(Conversation(
            id=web_sess_id, user_id=user_id, day_chat_id=dc_id,
            channel="web", is_active=True,
            started_at=base_ts, updated_at=base_ts,
        ))
        db.add(Conversation(
            id=tg_sess_id, user_id=user_id, day_chat_id=dc_id,
            channel="telegram", is_active=True,
            started_at=base_ts + timedelta(hours=2),
            updated_at=base_ts + timedelta(hours=2),
        ))
        await db.commit()

        # Lesson messages on web.
        for i, (role, content) in enumerate(LESSON):
            db.add(Message(
                id=f"msg-w-{i}-{uuid.uuid4().hex[:6]}",
                conversation_id=web_sess_id,
                day_chat_id=dc_id,
                role=role, content=content,
                created_at=base_ts + timedelta(minutes=i * 5),
            ))
        # Telegram chatter later in the day.
        for i, (role, content) in enumerate(TELEGRAM_NOISE):
            db.add(Message(
                id=f"msg-t-{i}-{uuid.uuid4().hex[:6]}",
                conversation_id=tg_sess_id,
                day_chat_id=dc_id,
                role=role, content=content,
                created_at=base_ts + timedelta(hours=2, minutes=i * 5),
            ))
        await db.commit()

    print(f"[seed] DayChat {dc_id[:8]} for user {user_id[:8]} on {yesterday_local}")
    return dc_id


async def force_archival(dc_id: str) -> bool:
    """Run the archival generator for this DayChat, synchronously."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from app.db.database import async_session_maker
    from app.db.models.day_chat import DayChat
    from app.services.day_summarizer import generate_archival_summary
    from sqlalchemy import select
    from datetime import datetime, timezone

    async with async_session_maker() as db:
        summary = await generate_archival_summary(db, dc_id)
        if not summary:
            print("[archival] FAILED to generate (likely missing API key)")
            return False
        dc = (await db.execute(select(DayChat).where(DayChat.id == dc_id))).scalar_one_or_none()
        if not dc:
            return False
        dc.archival_summary = summary
        dc.archival_summary_generated_at = datetime.now(timezone.utc)
        dc.archival_summary_status = "up_to_date"
        await db.commit()
    print(f"[archival] summary stored ({len(summary)} chars)")
    return True


# ── HTTP chat helpers (staging) ──────────────────────────────────────

async def chat_turn(base_url: str, token: str, message: str) -> dict:
    """POST a chat message to the staging agent and return the response."""
    import httpx

    async with httpx.AsyncClient(timeout=60, base_url=base_url) as client:
        r = await client.post(
            "/api/chat",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": message, "channel": "web"},
        )
        r.raise_for_status()
        return r.json()


async def login(base_url: str, email: str, password: str) -> str:
    import httpx

    async with httpx.AsyncClient(timeout=30, base_url=base_url) as client:
        r = await client.post(
            "/api/auth/login",
            json={"email": email, "password": password},
        )
        r.raise_for_status()
        return r.json().get("access_token") or r.json().get("token")


# ── Assertions on agent output ───────────────────────────────────────

REFUSAL_PHRASES = [
    "don't have a clear memory",
    "don't remember",
    "i can't recall",
    "no memory of",
    "trouble pulling up past conversations",
]


def _assert_grounded(response_text: str, must_contain_any: list[str], label: str):
    lower = response_text.lower()
    for bad in REFUSAL_PHRASES:
        if bad in lower:
            print(f"[FAIL:{label}] agent returned a refusal phrase: {bad!r}")
            print(f"  full response:\n{response_text[:1200]}")
            sys.exit(1)
    if not any(needle.lower() in lower for needle in must_contain_any):
        print(f"[FAIL:{label}] none of {must_contain_any} appeared in the response")
        print(f"  full response:\n{response_text[:1200]}")
        sys.exit(1)
    print(f"[ok:{label}] response contains expected content")


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", default=os.getenv("TARGET_USER_ID"))
    ap.add_argument("--user-tz", default=os.getenv("TARGET_USER_TZ", "UTC"))
    ap.add_argument("--base-url", default=os.getenv("STAGING_BASE_URL"))
    ap.add_argument("--email", default=os.getenv("STAGING_USER_EMAIL"))
    ap.add_argument("--password", default=os.getenv("STAGING_USER_PASSWORD"))
    ap.add_argument("--seed-only", action="store_true",
                    help="Seed yesterday + force archival, but skip HTTP chat assertions.")
    args = ap.parse_args()

    if not args.user_id:
        print("ERROR: TARGET_USER_ID (or --user-id) is required.")
        sys.exit(2)

    # Step 1-2: seed + archival (always; idempotent per-run).
    dc_id = await seed_yesterday(args.user_id, args.user_tz)
    archived = await force_archival(dc_id)
    if not archived:
        # Archival failure is non-fatal for the smoke test — the tool falls
        # back to the full conversation. But note it loudly.
        print("[WARN] archival did not complete; recall will fall back to raw messages.")

    if args.seed_only:
        print("\nseed-only: DONE. Go run the chat assertions manually if you want.")
        return

    if not (args.base_url and args.email and args.password):
        print("ERROR: STAGING_BASE_URL / STAGING_USER_EMAIL / STAGING_USER_PASSWORD required "
              "unless --seed-only is passed.")
        sys.exit(2)

    # Step 3: log in to staging.
    token = await login(args.base_url, args.email, args.password)
    print(f"[auth] logged in ({len(token)}-char token)")

    # Step 4: "do you remember yesterday?"
    print("\n[turn 1] do you remember what we talked about yesterday?")
    r1 = await chat_turn(args.base_url, token, "Do you remember what we talked about yesterday?")
    text1 = r1.get("text") or r1.get("content") or r1.get("response") or str(r1)
    print(text1[:1200])
    _assert_grounded(text1, [TOPIC, "derivative", "calculus"], "turn1_remember_yesterday")

    # Step 5: "make me a quiz about <topic>"
    print(f"\n[turn 2] make me a quiz from what you taught me yesterday about {TOPIC}")
    r2 = await chat_turn(args.base_url, token,
                          f"Make me a quiz from what you taught me yesterday about the {TOPIC}.")
    text2 = r2.get("text") or r2.get("content") or r2.get("response") or str(r2)
    print(text2[:2000])
    _assert_grounded(text2, [TOPIC, "d/dx", "derivative", "f'(g"], "turn2_quiz_grounded")

    print("\n✓ smoke test PASSED — agent recalls yesterday and builds a grounded quiz.")


if __name__ == "__main__":
    asyncio.run(main())
