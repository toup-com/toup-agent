"""Write-time quality gate for user memories.

Why this lives in CODE and not in the extraction prompt
-------------------------------------------------------
On 2026-07-30 the fleet had been running an extraction prompt containing

    9. **Do NOT extract general world knowledge.** Facts about companies,
       shows, products or public figures that would be true for everybody are
       NOT memories about this user ... If you would not be surprised to find
       it in an encyclopedia, skip it.

for TWELVE HOURS (image e62989b0, started 04:03 UTC) when the model wrote five
encyclopedia entries about 409A valuations into the founder's brain at 16:24.
The rule was live, correctly worded, unambiguous, and ignored — it sat between
"be THOROUGH — extract ALL of these if present" at the top of the prompt and
"Do NOT artificially limit — if there are 10 distinct facts, extract all 10" at
the bottom.

So every check here runs AFTER the model has spoken and is expressed as code,
which the model does not get a vote on. Nothing in this module asks an LLM
anything, and nothing in it costs a token.

Precision/recall was calibrated against the founder's live corpus (tenant
871bac24, 113 active rows hand-labelled 40 KEEP / 73 JUNK) — not guessed. Where
a rule costs a real memory, the cost is named in the rule's docstring.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional, Sequence, Set

# ── Thresholds, all calibrated against the live corpus ────────────────────
#
# The longest row a human judged worth keeping was 460 chars ("The user wants a
# short, useful daily Gmail briefing every day at 11:49 AM ..."). The longest
# junk row was 1002 chars — an entire lecture on stock options restated from
# the assistant's own answer and prefixed with "The user is interested in".
# 600 sits in the gap with ~30% headroom over the largest real memory.
#
# This is not a formatting preference. A memory is ONE standalone fact; a
# 1000-character multi-clause paragraph is a transcript, and storing it means
# the whole lecture is re-injected on every retrieval that touches it.
MAX_MEMORY_CHARS = 600

# A memory must be traceable to something the user said.
#
# This is a MARGIN, not two absolute thresholds. The first version required
# user-overlap <= 0.15 and failed on the exact case it was written for: the
# user asked "آپشن سهام چطور کار می‌کنه؟" and the answer restated "آپشن" and
# "سهام", so a pure-encyclopedia memory still scored 0.33 against the user's
# turn and the rule abstained. A question necessarily shares vocabulary with
# its answer; what separates an echo from a real fact is how much MORE of the
# memory lives in the assistant's words than in the user's.
ECHO_ASSISTANT_MIN = 0.60   # this much of the memory is in the assistant's answer
ECHO_MARGIN_MIN = 0.40      # ...and that much more than is in the user's
_ECHO_MIN_TOKENS = 4        # too few distinctive tokens to judge — stay quiet

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Tokens carried by nearly every memory ("The user wants...") — they say
# nothing about where the content came from, so they are excluded before any
# overlap ratio is computed. Deliberately tiny: a big stoplist would be a
# fourth vocabulary to drift (see memory_taxonomy's four-vocabulary problem).
_OVERLAP_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "been", "user", "users", "that", "this", "it", "its",
    "as", "at", "by", "with", "from", "their", "they", "them",
})

# How the user may refer to themselves, or be referred to, in a graph edge.
# Real display names are added per-call via `user_aliases`.
_SELF_REFERENTS = frozenset({
    "user", "the user", "me", "i", "my", "myself", "self", "owner",
})

# Entities that are the AGENT talking about itself. "Assistant summarizes
# Gmail" and "Assistant does not have reminder tool" are the agent's notes on
# its own plumbing, stored in the user's brain as if they were facts about the
# user's life.
_AGENT_SELF_NAMES = frozenset({
    "assistant", "the assistant", "agent", "the agent", "ai", "system", "bot",
})

# The sub-agent orchestrator names its workers "Sub-agent 1/2/3" in the parent
# reply; the extractor then reads that reply as user speech. Everything these
# match is the agent's own task decomposition.
_SCAFFOLDING_RE = re.compile(
    r"""(?ix)
    \b(?:
        sub[\s\-_]?agent(?:\s*\#?\d+)?     # Sub-agent 1, subagent 2, sub agent
      | tool[\s\-_]?call
    )\b
    """,
)

# Operational handles — "Job ID", "App ID", a bare UUID. As an ENTITY NAME
# these are never a fact about a person. Inside a longer sentence they may be
# incidental: the live corpus has a real project memory ("The user is working
# on a game app project named 'Build: Nokia Snake Arcade' with App ID <uuid>")
# that a content-wide match destroyed. So this is applied to relationship
# endpoints only, and _SCAFFOLDING_RE (which cannot fire incidentally) is what
# screens free text.
_OPERATIONAL_HANDLE_RE = re.compile(
    r"""(?ix)
    \b(?: job | app | task | run | build ) \s* id \b
    |
    \b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b
    """,
)

# Predicates that carry a standing user instruction even when neither end is
# the user. Kept SHORT and explicit: each entry is here because the live corpus
# contained a row worth keeping that the user-endpoint rule would otherwise
# have dropped.
# How much longer than its own triple a rendered row must be before it counts
# as "dedup merged real content into this". Measured on the live corpus: the
# two merged rows worth rescuing sit at 3.7x and 5.5x; every unmerged row is
# at ~1.0x because the renderer just splices the triple together.
_MERGE_ENRICHMENT_RATIO = 2.5

_STANDING_INSTRUCTION_PREDICATES = frozenset({
    "should_be_sent_via",
    "should_not_be_sent_via",
    "avoid_repeating_quotes_from",
    "avoids_repeating_quotes_from",
})


def _norm(text: Optional[str]) -> str:
    """Casefold + strip accents/punctuation noise for comparison only."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).casefold().strip()
    # Possessives: "Toup's launch" must compare equal to "Toup launch" so
    # `Toup --has_launch--> Toup's launch` is recognised as self-referential.
    text = re.sub(r"['’]s\b", "", text)
    return text


def _tokens(text: Optional[str]) -> Set[str]:
    return {t for t in _WORD_RE.findall(_norm(text)) if t}


def _content_tokens(text: Optional[str]) -> Set[str]:
    return _tokens(text) - _OVERLAP_STOPWORDS


def _overlap(needle: Set[str], haystack: Set[str]) -> float:
    """Fraction of `needle` present in `haystack` (0.0 when needle is empty)."""
    if not needle:
        return 0.0
    return len(needle & haystack) / len(needle)


# ── Relationship-edge gates ───────────────────────────────────────────────

def degenerate_relationship_reason(
    source: str, predicate: str, target: str
) -> Optional[str]:
    """Reject edges whose two ends do not carry independent information.

    `humanize_relationship` is a pure renderer with no semantic veto — it
    faithfully turns a worthless triple into a worthless, grammatical,
    permanently-embedded Memory row. These are the live rows it produced:

        video about context engeeniering --is_about--> context engeeniering
        LLM/AI researcher                --researches--> LLM/AI
        Toup                             --has_launch--> Toup's launch
        Gmail tool                       --connects_to--> Gmail
        Daily 5:06 PM ... quote routine  --includes--> motivational quote
        Top AI Papers This Week          --is_about--> AI
        OpenAI                           --is_subject_of--> latest OpenAI news

    Every one is caught by the containment test: after normalisation one end's
    content tokens are a subset of the other's, so the edge asserts only that a
    thing relates to itself.
    """
    s_raw, t_raw = _norm(source), _norm(target)
    if not s_raw or not t_raw:
        return "empty_endpoint"
    if s_raw == t_raw:
        return "self_edge"

    # Plain tokens, NOT _content_tokens: the overlap stoplist exists to stop
    # boilerplate skewing a RATIO, and it contains "user". Reusing it here made
    # `user --owns--> Toup` collapse to an empty source and get rejected as
    # contentless — measured against the live corpus, it was throwing away
    # "User owns Toup", "User works on mobile app" and "user uses Gmail".
    s_tok, t_tok = _tokens(source), _tokens(target)
    if not s_tok or not t_tok:
        return "no_content_tokens"
    if t_tok <= s_tok:
        return "target_restates_source"
    if s_tok <= t_tok:
        return "source_restates_target"
    return None


def scaffolding_reason(
    *texts: Optional[str], endpoints: bool = False
) -> Optional[str]:
    """Reject the agent's own task decomposition.

    Live rows: "Sub-agent 1 is researching Top AI Papers This Week",
    "Sub-agent 2 summarizes Latest OpenAI News". The orchestrator correctly
    passes disable_post_processing=True, but the parent turn's reply NAMES its
    workers, so the parent's own extraction reads them back as user facts.

    `endpoints=True` additionally rejects operational handles, and is used only
    when the strings are entity NAMES rather than prose — see
    _OPERATIONAL_HANDLE_RE for why that distinction matters.
    """
    for text in texts:
        if not text:
            continue
        if _SCAFFOLDING_RE.search(str(text)):
            return "agent_scaffolding"
        if endpoints and _OPERATIONAL_HANDLE_RE.search(str(text)):
            return "operational_identifier"
    return None


def _is_user_endpoint(name: Optional[str], aliases: Set[str]) -> bool:
    n = _norm(name)
    if not n:
        return False
    if n in _SELF_REFERENTS or n in aliases:
        return True
    # "the user's brother" / "user account" — a possessive reference still puts
    # the user on this end of the edge.
    first = n.split()[0] if n.split() else ""
    return first in _SELF_REFERENTS or first in aliases


def relationship_gate_reason(
    source: str,
    predicate: str,
    target: str,
    *,
    user_aliases: Optional[Iterable[str]] = None,
    rendered: Optional[str] = None,
) -> Optional[str]:
    """Decide whether a graph edge deserves to become a user-visible Memory.

    The knowledge graph itself (`entity_relationships`) is NOT gated — it keeps
    every edge, so traversal and the entity map lose nothing. This governs only
    the Memory MIRROR: what the user reads on the Memory screen and what is
    embedded into the retrieval corpus.

    The rule the extraction prompt already states in prose ("The USER, or
    someone/something in the USER's life, must be one end of the edge") is
    applied here as code, because in production the prompt version let through
    "Better Call Saul is available on Netflix", "Drake artist of 0-100",
    "Shops at Don Mills is in Toronto" and "Codex is included in ChatGPT Free".

    MEASURED against the 113 hand-labelled live rows (backend/tests/
    test_memory_junk_gate.py replays the corpus, so these numbers are asserted,
    not remembered):

        tautology         8/8   rejected
        scaffolding       7/7   rejected
        world_knowledge  15/15  rejected
        KEEP retained    38/40  (95%)

    It abstains on three other junk classes ON PURPOSE, because each has a
    correct fix elsewhere and catching them here would need brittle
    English-only phrase matching:
        routine_internals -> disable_post_processing on the routine handler
        one_shot_request  -> ttl_days passthrough on the voice tunnel
        degenerate_or_dup -> dedup candidate-selection fix

    The two KEEP rows it costs are both genuinely marginal, and neither loses
    information the brain holds nowhere else:
      - "Gmail may require re-authentication" — generic, and the actionable
        version survives in the conversation-path briefing spec.
      - "LLM/AI researcher is affiliated with University of Toronto" — the
        subject is not a real person, it is an occupation string the extractor
        invented as an entity.
    """
    reason = scaffolding_reason(source, target, endpoints=True)
    if reason:
        return reason

    reason = degenerate_relationship_reason(source, predicate, target)
    if reason:
        return reason

    if _norm(source) in _AGENT_SELF_NAMES or _norm(target) in _AGENT_SELF_NAMES:
        return "agent_talking_about_itself"

    key = _norm(predicate).replace(" ", "_")
    if key in _STANDING_INSTRUCTION_PREDICATES:
        return None

    aliases = {_norm(a) for a in (user_aliases or ()) if _norm(a)}
    if _is_user_endpoint(source, aliases) or _is_user_endpoint(target, aliases):
        return None

    # Dedup MERGES relationship rows: the row keeps its original (now stale)
    # triple in metadata while its prose grows into something much richer.
    # `Gmail --reconnect_using--> inline connector card` became a 235-char
    # standing instruction that names the user's daily briefing. Judging that
    # row on its stale triple alone throws away real work.
    #
    # The rescue keys on ENRICHMENT, not on the word "user": requiring only a
    # self-referent let 12 junk rows back in, because routine-internals rows
    # ("Gmail briefing summarizes email for user") mention the user too. A
    # merged row is several times longer than its own triple; an unmerged one
    # is the same length as it, by construction.
    if rendered and _MERGE_ENRICHMENT_RATIO > 0:
        triple_len = len(f"{source} {predicate} {target}".strip())
        if triple_len and len(rendered.strip()) >= triple_len * _MERGE_ENRICHMENT_RATIO:
            return None

    return "no_user_endpoint"


# ── Extracted-memory gates ────────────────────────────────────────────────

def assistant_echo_reason(
    content: str,
    user_message: Optional[str],
    assistant_response: Optional[str],
) -> Optional[str]:
    """Reject memories lifted from the assistant's own answer.

    The extractor prompt interpolates the assistant's COMPLETE reply as
    extraction material, so a 3000-token lecture occupies most of the prompt's
    token mass and the model dutifully mines it. Rule 3 ("Only extract
    information STATED BY THE USER") is prose and does not bind.

    Fires only when the content is substantially present in the assistant's
    reply AND essentially absent from the user's. That conjunction is what
    keeps it quiet in the cross-lingual case this user actually hits: a memory
    written in English from a Persian voice turn overlaps neither side, so the
    rule abstains rather than deleting a real fact.
    """
    tokens = _content_tokens(content)
    if len(tokens) < _ECHO_MIN_TOKENS:
        return None
    if not assistant_response:
        return None

    from_assistant = _overlap(tokens, _content_tokens(assistant_response))
    from_user = _overlap(tokens, _content_tokens(user_message))

    if (
        from_assistant >= ECHO_ASSISTANT_MIN
        and (from_assistant - from_user) >= ECHO_MARGIN_MIN
    ):
        return "assistant_echo"
    return None


def memory_gate_reason(
    content: str,
    *,
    user_message: Optional[str] = None,
    assistant_response: Optional[str] = None,
) -> Optional[str]:
    """Full write-time screen for one extracted memory. None == store it."""
    text = (content or "").strip()
    if not text:
        return "empty"
    if len(text) > MAX_MEMORY_CHARS:
        return "not_a_single_fact"

    reason = scaffolding_reason(text)
    if reason:
        return reason

    return assistant_echo_reason(text, user_message, assistant_response)


__all__ = [
    "MAX_MEMORY_CHARS",
    "ECHO_ASSISTANT_MIN",
    "ECHO_USER_MAX",
    "assistant_echo_reason",
    "degenerate_relationship_reason",
    "memory_gate_reason",
    "relationship_gate_reason",
    "scaffolding_reason",
]
