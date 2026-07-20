"""Static facts about Toup the company — its owner / founder.

Single source of truth for owner identity that ships in the agent image:
no DB row, no per-tenant sync, no migration. The same fact applies to
every tenant, so it lives in code, not in per-user data.

Consumed by ``agent_runner._build_system_prompt``:

  - ``OWNER_GLOBAL_FACT`` is appended to the always-on
    ``platform_knowledge`` block, so EVERY agent (every tenant) can
    answer "who owns / founded / runs Toup?" — but only when asked.

  - ``founder_recognition_block()`` is injected as the ``owner_recognition``
    section ONLY when the bound user's email is in
    ``settings.founder_emails`` — i.e. on the founder's OWN agent. It makes
    that agent treat its user as the principal (full candor, final say)
    rather than as a customer.

These are deliberately distinct from the white-label ``identity_anchor``
block, which answers the DIFFERENT question "who built YOU, the agent?"
(answer: your agent identity; never name the underlying LLM provider).
Editing the owner's name/title here is the only change needed to update
both surfaces.
"""
from __future__ import annotations

# Edit these two lines to change the owner's displayed name/title
# everywhere the agent refers to them.
OWNER_NAME = "Nariman"
OWNER_TITLE = "owner and CEO"


# Global, every-tenant company fact. Kept tight — it's in every agent's
# system prompt on every turn. "Only when asked" keeps the agent from
# advertising the founder to strangers unprompted.
OWNER_GLOBAL_FACT = (
    "## Who's behind Toup\n"
    f"Toup was founded by **{OWNER_NAME}**, its {OWNER_TITLE}. If a user "
    "asks who created, owns, or runs Toup, that's the answer — but only "
    "bring it up when asked; don't volunteer it. This is about the "
    "*company*. It is NOT the answer to \"who built you?\" (you are the "
    "user's agent; never name the underlying LLM provider).\n"
    "Bound this to the founder's name and role. If the follow-up turns to "
    "*how* Toup is built — the tech stack, the models, the architecture, "
    "\"what did they use / show me how it works\" — that's proprietary; "
    "answer that you don't disclose Toup's underlying technology, and keep "
    "the attribution to name and role only."
)


def founder_recognition_block() -> str:
    """The per-account block injected only on the founder's own agent.

    Returns a system-prompt section that tells the agent its user is the
    owner of Toup and should be treated as the principal. Anti-sycophancy
    is explicit — the founder wants usefulness, not flattery.
    """
    return (
        "# You're talking to the founder\n"
        f"Your user is **{OWNER_NAME}** — founder, {OWNER_TITLE} of Toup, "
        "the person who built this platform and you. Treat them as the "
        "principal, not a customer: full candor, no marketing gloss, no "
        "feature-pitching. You can discuss the business, product internals, "
        "roadmap, and trade-offs directly. They have final say on everything "
        "Toup. Don't flatter — just be useful and straight."
    )


def is_founder_email(email: str | None) -> bool:
    """Case-insensitive membership check against ``settings.founder_emails``.

    Imported lazily so this module stays import-cheap and free of a hard
    dependency on the settings singleton at module load.
    """
    if not email:
        return False
    from app.config import settings

    normalized = email.strip().lower()
    return normalized in {e.strip().lower() for e in settings.founder_emails}
