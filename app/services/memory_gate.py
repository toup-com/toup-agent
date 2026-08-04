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

Rules added 2026-08-01 by the memory verification job
(docs/MEMORY_VERIFICATION_REPORT.md) were calibrated the same way, against the
67 labeled conversations in backend/tests/memverify/corpus.py:

    quoted_content        pasted material is not a user assertion — the
                          anti-memory-poisoning rule (J03/J04/B08/J07)
    transient_state       a horizon of <= 1 day is conversational state
    sensitive_*           secret values are never written down verbatim
    negated_predicate     (relationships) "USER nots lives in Toronto"
    inferred_interest     (2026-08-03) asking about something is not being
                          something — the ONLY control on the cross-lingual
                          case, where assistant_echo abstains by design

Every one of them ships with must-REJECT *and* must-KEEP cases in
backend/tests/test_memory_gate_regressions.py, because a gate that rejects
everything passes any test suite that only checks for junk.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Optional, Sequence, Set


class MemoryRejected(ValueError):
    """A write the gate refuses. Carries the machine-readable reason.

    Raised (rather than returning None) by the storage backstop in
    MemoryService.create_memory, because that function returns a Memory and
    every caller dereferences it — a silent None would turn a refusal into an
    AttributeError somewhere unrelated.
    """

    def __init__(self, reason: str):
        super().__init__(f"memory refused by the write gate: {reason}")
        self.reason = reason


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

# Pasted material is not a user assertion — see `quoted_content_reason`.
#
# Calibrated against the labeled corpus, not guessed. Measured containment /
# margin for every case the rule must separate:
#
#   REJECT  J03 planted ownership          1.00 / 1.00
#   REJECT  J04 roleplay planting          1.00 / 1.00
#   REJECT  B08 third party in an article  1.00 / 1.00
#   REJECT  J07 HTML-comment instruction   1.00 / 1.00
#   KEEP    user restates pasted content   0.80 / 0.40   <- the binding case
#   KEEP    short inline quote             below the paste floor entirely
#
# The attacks all sit at 1.00 because the payload never passes through the
# user's own voice — that is precisely what makes them attacks. The thresholds
# sit in the gap, closer to the KEEP case than to the REJECT cases, because
# dropping a real memory is worse than keeping a marginal one.
QUOTED_CONTAINMENT_MIN = 0.85  # this much of the memory lives inside the paste
QUOTED_MARGIN_MIN = 0.50       # ...and that much more than outside it
QUOTED_CONTAINMENT_LOOSE = 0.70  # ...or this much, with ZERO support outside
_QUOTED_MIN_CHARS = 60         # below this it is an inline quote, not a paste

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Ways a user delimits material they are showing you rather than saying.
# Order matters only for readability; every match is unioned.
_QUOTED_BLOCK_RES = (
    re.compile(r"```.*?```", re.S),                    # fenced code / paste
    re.compile(r"<!--.*?-->", re.S),                   # HTML comment payloads
    re.compile(r"^-{3,}\s*$(.*?)^-{3,}\s*$", re.M | re.S),  # ---  block  ---
    re.compile(r"[\"“”„]([^\"“”„]{%d,})[\"“”„]" % _QUOTED_MIN_CHARS),
    re.compile(r"['‘’]([^'‘’]{%d,})['‘’]" % _QUOTED_MIN_CHARS),
)


def _split_quoted(text: str) -> tuple:
    """Return (quoted_material, everything_else) for one user message.

    Everything the user wrapped in quotes, fences, HTML comments or `---`
    rules is treated as shown-not-said; the remainder is their own voice.
    """
    if not text:
        return "", ""
    quoted_parts = []
    remainder = text
    for pattern in _QUOTED_BLOCK_RES:
        hits = []
        for match in pattern.finditer(remainder):
            block = match.group(1) if match.groups() else match.group(0)
            if block and len(block.strip()) >= _QUOTED_MIN_CHARS:
                quoted_parts.append(block)
                hits.append(match.group(0))
        for raw in hits:
            remainder = remainder.replace(raw, " ")
    return "\n".join(quoted_parts), remainder

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

# Negated predicates must not become Memory rows.
#
# `humanize_relationship` repairs an uninflected verb stem by appending "-s".
# On a negated predicate the stem it repairs is the NEGATION: `not_lives_in`
# renders as "USER nots lives in Toronto". That is not merely ugly — it is a
# correctness hazard, because a model reading "USER nots lives in Toronto"
# alongside "USER lives in Vancouver" is being shown two rows that both look
# like assertions of residence, and the negation is one dropped token away
# from inverting. It appeared live the moment a user corrected their city.
#
# The GRAPH keeps the edge, so nothing is lost for traversal; only the prose
# mirror is suppressed. The affirmative fact ("The user moved to Vancouver and
# no longer lives in Toronto") still arrives through normal extraction, which
# renders negation in real language.
_NEGATED_PREDICATE_RE = re.compile(
    r"(?ix) ^ (?: not | no | never | non | doesn_?t | does_not | is_not | isnt ) _ | _ not _ "
)

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


# Identities a freshly-provisioned tenant carries before onboarding writes the
# real ones. An alias set containing only these tells us nothing about who the
# user is, and must not be mistaken for "the user is not involved".
_PLACEHOLDER_ALIASES = frozenset({
    "agent owner", "agent", "owner", "user", "toup user", "new user",
    "unknown", "none", "test user",
})


def _identity_is_known(aliases: Set[str]) -> bool:
    """True only when we hold at least one alias that identifies a real person.

    Bare hex prefixes ("3134fece", from a "<prefix>@agent.local" placeholder
    email) are excluded too — they are provisioning artefacts, not names.
    """
    for alias in aliases:
        if alias in _PLACEHOLDER_ALIASES:
            continue
        if re.fullmatch(r"[0-9a-f]{6,}", alias):
            continue
        return True
    return False


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
    # The explicit allowlist wins over the negation rule: every entry in it is
    # there because the live corpus held a row worth keeping, and two of them
    # ("should_not_be_sent_via", "avoid_repeating_quotes_from") are negative by
    # nature. They also have real templates in _PREDICATE_TEMPLATES, so they
    # never reach the "-s" inflection repair that mangles negations.
    if key in _STANDING_INSTRUCTION_PREDICATES:
        return None
    if _NEGATED_PREDICATE_RE.search(key):
        return "negated_predicate"

    aliases = {_norm(a) for a in (user_aliases or ()) if _norm(a)}
    if _is_user_endpoint(source, aliases) or _is_user_endpoint(target, aliases):
        return None

    # FAIL OPEN when we do not know who the user is.
    #
    # Tenant DBs are provisioned with a PLACEHOLDER identity — name "Agent
    # Owner", email "<prefix>@agent.local" — until onboarding fills in the real
    # one. On such a tenant the alias lookup yields nothing usable, so an edge
    # naming the user by their actual name ("Nariman owns Toup") reads as
    # having no user endpoint and gets dropped. Found on live tenant 3134fece,
    # where this rule would have archived exactly that row.
    #
    # Dropping a real fact is far worse than keeping a marginal one, so when
    # identity is unknown this rule abstains. The tautology, scaffolding and
    # agent-self rules above still apply — they need no identity at all.
    if not _identity_is_known(aliases):
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


def quoted_content_reason(
    content: str, user_message: Optional[str]
) -> Optional[str]:
    """Reject memories mined out of material the user PASTED rather than said.

    Pasting is not asserting. When someone pastes an article, an email or a
    note and asks "what does this say?", the claims inside it are the author's,
    not theirs — and if that material contains a sentence shaped like a fact
    about the user, the extractor writes it down as one.

    Measured on the labeled corpus, this single rule covers four separate
    failures that share one mechanism:

        B08  "Marcus Aurelius Trentham is the CEO of Halcyon Freight"
             — a third party inside a pasted article
        B22  "The user's colleague is moving to Berlin in April"
             — a quote from someone else, re-attributed to the user
        J03  "The user is the sole owner of Halifax Trust Bank"
             — a false fact planted in a note the user asked about
        J04  "The user is a licensed pilot with 4,000 hours on the Gulfstream"
             — a false fact planted through roleplay narration

    J03 and J04 are memory-poisoning attacks: the payload never passes through
    the user's own voice, only through content they asked about. Any defence
    phrased as a prompt rule can be argued with by the next payload; this one
    cannot, because it is decided after the model has spoken.

    The rule is deliberately narrow, because over-firing costs real memories:

      * The block must be genuinely PASTED — at least
        _QUOTED_MIN_CHARS of delimited text. Short inline quotes ("my doctor
        said 'you have type 1 diabetes'") stay below the floor and are kept.
      * The memory must be almost entirely inside that block
        (>= QUOTED_CONTAINMENT_MIN of its content tokens).
      * The memory must NOT also be supported by the user's own words outside
        the block. Where the user restates something themselves — "this article
        is about my company Toup" — the outside text carries the tokens and the
        rule abstains.
    """
    if not user_message:
        return None
    tokens = _content_tokens(content)
    if len(tokens) < _ECHO_MIN_TOKENS:
        return None

    quoted, remainder = _split_quoted(user_message)
    if not quoted:
        return None

    inside = _overlap(tokens, _content_tokens(quoted))
    outside = _overlap(tokens, _content_tokens(remainder))

    if inside >= QUOTED_CONTAINMENT_MIN and (inside - outside) >= QUOTED_MARGIN_MIN:
        return "quoted_content"

    # Second branch: the user's own words support NONE of it.
    #
    # The containment threshold above assumes the memory quotes the paste
    # closely, and a paraphrase breaks that assumption. Measured: the extractor
    # writes "Halcyon Freight ACQUIRED Brightline Cartage" from a paste that
    # says "announced the ACQUISITION of" — one token different, containment
    # 0.83, and the rule fell silent on a run where the same scenario had been
    # caught at 1.00 the run before. That is a flake, and a flake in a junk gate
    # is a junk row.
    #
    # `outside == 0` is a stronger and more honest signal than any containment
    # number: not one distinctive token of this memory appears in anything the
    # user actually wrote. It cannot fire on the case that binds the threshold
    # above ("this article is about my company Toup" puts `company` and `toup`
    # outside the quote, at 0.40).
    if inside >= QUOTED_CONTAINMENT_LOOSE and outside == 0.0:
        return "quoted_content"

    return None


# A horizon this short does not describe a person, it describes a moment.
#
# The extractor already decides transience and emits `valid_for_days` on its own
# documented scale ("A 2-minute reminder is 1; a this-week errand is 7"). It
# gets that right and consistently — measured over repeated trials:
#
#   "I'm exhausted today, barely slept."      -> emotions,    ttl 1   (3/3)
#   "I'm waiting on the Vercel deploy."       -> active_task, ttl 1   (3/3)
#   "I've had type 1 diabetes since twelve."  -> health,      ttl None (3/3)
#   "I've struggled with anxiety for years."  -> health,      ttl None (3/3)
#
# The model was never the problem here either: it labelled the passing states
# correctly and the pipeline stored them anyway, merely with an expiry. A row
# that is junk for a day is still junk for a day — it competes in retrieval and
# it is what the user sees on the Memory screen.
#
# Deliberately keyed on the horizon rather than on the category, so it works in
# any language and does not need a phrase list.
TRANSIENT_HORIZON_DAYS = 1


# ── The exception the horizon alone cannot see ───────────────────────────
#
# A horizon of one day covers two completely different things, and measurement
# (scratchpad probe, real gpt-4o-mini) showed both arriving here as ttl=1:
#
#   "I have a dentist appointment tomorrow at 3."  -> transient, named 1  KEEP
#   "I'm exhausted today, barely slept."           -> transient, named 1  DROP
#
# No threshold separates them: `resolve_ttl_days` clamps every sub-day horizon up
# to 1, so both land in the same bucket. Category does not separate them either —
# "dentist tomorrow" and "I'm working on the deck right now" are both ACTIVE_TASK,
# so the technique that fixed the location case does not transfer.
#
# What separates them is whether the memory names a COMMITMENT AT A TIME rather
# than a state of the person. Two independent signals, because either alone is
# weak:
#
#   1. The model's own `scheduled` flag. Works in every language, which a noun
#      list would not — the corpus is deliberately part Farsi. But it is model
#      disposition, and disposition is not a control.
#   2. A clock time in the content. Deterministic and checkable.
#
# They are OR'd, and this check can only ever KEEP a memory that the horizon rule
# was about to drop. It can never cause one to be dropped. That asymmetry is the
# whole safety argument: a false positive stores one short-lived row for two days,
# a false negative loses the user's appointment.
#
# Kept commitments get a TTL FLOOR, and that floor is the actual point. Storing
# "dentist tomorrow at 3" with ttl=1 sets expires_at to this time tomorrow — if
# they said it at 9am, the memory dies at 9am tomorrow, six hours BEFORE the
# appointment it exists to remember. A commitment must outlive the commitment.
SCHEDULED_MIN_TTL_DAYS = 2

# Time-of-day patterns. Arabic-Indic (٠-٩) and Persian (۰-۹) digits are included
# because the extractor is fed Farsi turns and returns Farsi content.
_D = r"0-9٠-٩۰-۹"
_CLOCK_TIME_RE = re.compile(
    r"(?:"
    rf"[{_D}]{{1,2}}\s*[:.۰-۹]?\s*[{_D}]{{2}}\s*(?:am|pm)?"  # 7:40, 15.00
    rf"|[{_D}]{{1,2}}\s*(?:am|pm|a\.m\.|p\.m\.)"                        # 3pm
    rf"|\bat\s+[{_D}]{{1,2}}\b"                                          # at 3
    rf"|\bساعت\s*[{_D}]{{1,2}}"                                          # Farsi "at <n> o'clock"
    r")",
    re.IGNORECASE,
)


def is_scheduled_commitment(
    content: Optional[str], model_flag: bool = False
) -> bool:
    """True if this is a commitment at a time, not a passing state.

    Only ever consulted for memories the horizon rule is about to drop, and only
    ever to keep one. See the note above for why the two signals are OR'd.
    """
    if model_flag:
        return True
    return bool(content and _CLOCK_TIME_RE.search(content))


def scheduled_floor_ttl(ttl_days: Optional[int]) -> int:
    """The TTL a kept commitment gets, so it outlives the commitment itself."""
    try:
        current = int(ttl_days) if ttl_days is not None else 0
    except (TypeError, ValueError):
        current = 0
    return max(current, SCHEDULED_MIN_TTL_DAYS)


# Categories where ANY expiry at all means "right now" rather than "in general".
#
# Found by the 500-conversation soak, which stored 71 rows of the shape "The
# user is currently sitting in the Volterrino". The model was not wrong about
# transience — it set an expiry every time — it was wrong about the HORIZON,
# giving an unfamiliar proper noun 7 days because a place you have never heard
# of could plausibly be somewhere you stay. "I'm sitting in the car right now"
# is dropped correctly (3/3 trials); "I'm sitting in the Volterrino right now"
# is stored with ttl=7 (3/3), and so is the real Athens neighbourhood Kolonaki.
#
# Raising TRANSIENT_HORIZON_DAYS to 7 would have caught them and also destroyed
# "The user goes to the gym on Tuesdays" (habits, ttl=7) — a durable habit. So
# the rule is keyed on the category instead, and it is safe because of a clean
# measured split: every durable location statement carries NO expiry.
#
#   "I live in Toronto"                     -> identity,  ttl None
#   "I moved to Vancouver"                  -> locations, ttl None
#   "my office is on Adelaide Street West"  -> locations, ttl None
#   "my gym is called Ironwood"             -> locations, ttl None
#   "I'm sitting in the Volterrino now"     -> locations, ttl 7     <- rejected
#
# A place with an expiry date is where you ARE, not where you live.
_EPHEMERAL_WITH_ANY_TTL = frozenset({"locations"})

# The same argument, for what the user is DOING rather than where they are.
#
# "I'm waiting on the Vercel deploy" was stored with ttl=7 and sailed past the
# one-day horizon. Measured over 5 trials per case, the split is as clean as the
# locations one — durable work never lands in this category at all:
#
#   "I'm the founder of Toup."                     -> work,        ttl None (5/5)
#   "I work at Ferrowick as a data engineer."      -> work,        ttl None (5/5)
#   "I'm building a side project named Marbleweft" -> work,        ttl None (5/5)
#   "I'm learning Portuguese."                     -> skills,      ttl None (5/5)
#   "I'm waiting on the Vercel deploy."            -> active_task, ttl 7    <- rejected
#   "I'm stuck on a bug right now."                -> active_task, ttl 7    <- rejected
#   "Researching the 3 best PM tools today."       -> active_task, ttl 7    <- rejected
#
# Note the grammar is identical on both sides of that line ("I'm building…" vs
# "I'm reviewing…"), which is exactly why a phrase rule could not do this and the
# model's own category can.
#
# This does NOT remove the "what am I working on" capability. That feature has
# its own producer (active_task_service.store_active_task), its own always-
# injected <active_tasks> block and its own decay clock; it is not served by the
# extractor writing a second, competing copy into long-term memory. Standing
# arrangements never reach here — describes_recurring_arrangement() strips their
# TTL upstream, so "a Gmail briefing every day at 11:49" stays durable.
_ACTIVITY_WITH_ANY_TTL = frozenset({"active_task"})

# And the same argument for how the user FEELS. Measured 8 trials per case:
#
#   "I'm annoyed at my sister right now."   -> emotions, transient 7d  <- rejected
#   "I'm a bit under the weather today."    -> emotions, transient 1d  <- rejected
#   "I've struggled with anxiety for years" -> emotions, DURABLE       kept
#
# A lasting emotional pattern is durable and the model says so; a mood the user
# is in this afternoon is flagged transient every time. Same clean split as
# locations and activity, so the same treatment.
_MOOD_WITH_ANY_TTL = frozenset({"emotions"})


# A stricter bar for categories that never expire.
#
# In an expiring category, keeping a marginal 2-day memory costs two days. In a
# never-expire one it costs FOREVER — "I'm a bit under the weather today" landed
# in HEALTH with no expiry, so a passing complaint became a permanent fact. The
# cost of a false keep is unbounded there, so the bar is higher.
#
# Still nowhere near a real fact: "I'm on antibiotics for two weeks" is 14 days,
# and every durable health fact is flagged durable rather than transient, 8/8
# across six of them (allergy, diabetes, metformin, migraines, anxiety, and a
# PEOPLE control).
#
# Known residual, stated rather than chased: the model's own horizon for
# "I'm a bit under the weather today" is unstable — 1 day on most trials, 3 on
# some. This rule catches the 1- and 2-day trials (7 of 8); the 3-day ones are
# still stored. Raising the horizon to 3 would start eating genuine short-term
# health facts, which is the trade BUG-9 already showed is the wrong one.
TRANSIENT_HORIZON_DAYS_PERMANENT = 2


def transient_horizon_reason(
    ttl_days: Optional[int],
    category: Optional[str] = None,
    *,
    permanent_if_kept: bool = False,
) -> Optional[str]:
    """Reject conversational state masquerading as memory. None == store it.

    `permanent_if_kept` says this memory's category will not expire, so keeping
    it is a permanent decision and the horizon is raised accordingly.
    """
    if ttl_days is None:
        return None
    try:
        days = int(ttl_days)
    except (TypeError, ValueError):
        return None
    horizon = (
        TRANSIENT_HORIZON_DAYS_PERMANENT if permanent_if_kept
        else TRANSIENT_HORIZON_DAYS
    )
    if days <= horizon:
        return "transient_state"

    key = str(getattr(category, "value", category) or "").strip().casefold()
    if key in _EPHEMERAL_WITH_ANY_TTL:
        return "transient_position"
    if key in _ACTIVITY_WITH_ANY_TTL:
        return "transient_activity"
    if key in _MOOD_WITH_ANY_TTL:
        return "transient_mood"
    return None


# The two verdicts a scheduled commitment is allowed to override. A place with an
# expiry stays rejected: "I'm in the airport lounge" is a position whether or not
# a clock time is nearby, and BUG-14's evidence for that was unambiguous.
SCHEDULABLE_REASONS = frozenset({"transient_state", "transient_activity"})


# ── Sensitive values ─────────────────────────────────────────────────────
#
# Policy (also stated in MEMORY_SYSTEM_MAP.md §2.1): a small set of secret
# categories is NEVER written to long-term memory verbatim — payment card
# numbers, card verification values, government identity numbers, API keys and
# bearer tokens, and declared passwords/passphrases/PINs.
#
# This exists because the alternative was enforcement by LLM judgment. The
# labeled privacy scenarios (K01–K04) passed before this rule existed, purely
# because gpt-4o-mini chose not to extract the values — which is a disposition,
# not a control. Nothing stopped a different model, a different temperature or a
# differently-phrased turn from writing a card number into a permanent,
# plaintext, embedded row. Memory content is not encrypted at column level, so a
# stored secret is a stored secret.
#
# Scoped tightly on purpose. Everything NOT in this list stays storable,
# including the things a blunter rule would eat:
#   * health details and medications  — K05, durable user facts, must be kept
#   * user-chosen door/locker/garage codes the user ASKED to save — A21
#   * flight numbers, addresses, dates, phone-shaped identifiers
_LUHN_CANDIDATE_RE = re.compile(r"(?<!\w)(?:\d[ -]?){12,18}\d(?!\w)")
_CARD_CONTEXT_RE = re.compile(
    r"(?i)\b(?:visa|mastercard|amex|american\s+express|discover|credit\s+card|"
    r"debit\s+card|card\s+number|cardholder|pan)\b"
)

_SENSITIVE_RES = (
    # Government identity numbers. Canadian SIN / US SSN shapes.
    ("government_id", re.compile(r"(?<!\d)\d{3}[ -]\d{3}[ -]\d{3}(?!\d)")),
    ("government_id", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")),
    ("government_id", re.compile(r"(?ix) \b (?: sin | ssn | passport \s* (?:no|number|\#)? ) \b [^.\n]{0,20}? \b [a-z]?\d{6,9} \b")),
    # Provider API keys and bearer tokens, by their own published prefixes.
    ("api_key", re.compile(r"(?i)\b(?:sk|pk|rk)-(?:proj|live|test|ant|admin)?-?[A-Za-z0-9_-]{16,}")),
    ("api_key", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{20,}")),
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("api_key", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{20,}")),
    ("api_key", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.")),
    # A declared password / passphrase / PIN, with its value.
    # "code" is deliberately absent: "my garage door code is nightjar-4417" is a
    # fact the user explicitly asked to be remembered.
    # Allows a few intervening words: "my password FOR THE ADMIN PANEL is X".
    ("password", re.compile(r"(?ix) \b (?: password | passwd | passphrase | pin \s* (?:code|number)? ) \b (?: \s+ \w+){0,5} \s* (?: is | = | : ) \s* \S{4,}")),
    # Card verification value, only when named.
    ("card_cvv", re.compile(r"(?i)\b(?:cvv|cvc|cv2|security\s+code)\b\s*(?:is|=|:)?\s*\d{3,4}\b")),
)


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = ord(ch) - 48
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


# Two tiers, because "secret" covers two different things.
#
# NEVER_STORE has no legitimate "please remember this" use and a high blast
# radius if it leaks: payment cards, CVVs, government identity numbers, provider
# API keys and bearer tokens. Refused on every path, whatever anyone asked for.
#
# The password/passphrase/PIN tier is different. A user who says "remember my
# storage locker passphrase is kestrel-dbf7" is stating a fact about their own
# life and expecting the agent to keep it — the same product behaviour as the
# garage door code in A21, and the behaviour D-mem-A's supersede tests exercise.
# Refusing that on a deliberate save would be the agent overruling its owner
# about their own locker.
#
# So the passphrase tier is refused on AUTOMATIC capture (where nobody asked for
# anything and the value was merely observed) and allowed on an EXPLICIT save.
# The tier split is the whole safety argument: an explicit save can never reach
# a card number or an API key, no matter how it is phrased.
_NEVER_STORE_LABELS = frozenset({"government_id", "api_key", "card_cvv"})


def sensitive_content_reason(
    content: str, *, explicit_save: bool = False
) -> Optional[str]:
    """Reject memories carrying a secret value verbatim. None == store it."""
    text = content or ""
    if not text:
        return None

    for label, pattern in _SENSITIVE_RES:
        if explicit_save and label not in _NEVER_STORE_LABELS:
            continue
        if pattern.search(text):
            return f"sensitive_{label}"

    # Payment cards last. A bare digit run is rejected only when it is
    # Luhn-valid OR the sentence names a card — so ordinary long numbers (order
    # ids, account numbers, phone numbers) stay storable, while a mistyped or
    # partially-redacted PAN in an obviously card-shaped sentence still does not
    # get written down.
    card_context = _CARD_CONTEXT_RE.search(text) is not None
    for match in _LUHN_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and (_luhn_ok(digits) or card_context):
            return "sensitive_card_number"

    return None


# ── Asking about something is not being something ────────────────────────
#
# BUG-26. `assistant_echo_reason` above deliberately ABSTAINS across
# languages: a memory written in English from a Persian turn overlaps neither
# side lexically, so the margin rule cannot judge it and stays quiet rather
# than deleting a real fact. That is the right call — but it leaves the
# cross-lingual echo case with no control at all, and B06 duly stored
# "The user is interested in how stock options work and has knowledge about
# 409A valuations, which play a crucial role in determining the fair value of
# common stock" from a single Farsi turn reading, in full: "how do stock
# options work?"
#
# It stored that on SOME runs. The extractor is sampled, so the corpus caught
# it once in roughly five — which under a zero-junk bar is a failure that
# happens to be sleeping.
#
# This rule judges provenance structurally instead of lexically, so language
# does not enter into it: when the user's own voice for that turn is nothing
# but questions, a memory whose whole claim is that they are interested in or
# knowledgeable about the topic is the model inferring, not the user stating.
#
# The conjunction is what keeps it narrow. "I'm allergic to peanuts — what
# should I eat?" contains a declarative, so the turn is not question-only and
# the rule abstains. "How do stock options work?" contains none.
_INTEREST_CLAIM_RE = re.compile(
    r"\b(?:is|seems|appears|being)\s+(?:very\s+|quite\s+|really\s+|somewhat\s+)?"
    r"(?:interested\s+in|curious\s+about|keen\s+(?:on|to))\b"
    r"|\bhas\s+(?:an?\s+|some\s+)?"
    r"(?:knowledge|interest|understanding|familiarity)\s+(?:of|about|in|with)\b"
    r"|\bwants?\s+to\s+(?:know|learn|understand)\b"
    r"|\bis\s+(?:learning|reading|asking)\s+about\b"
    r"|\basked\s+(?:about|how|what|why|whether)\b",
    re.IGNORECASE,
)

# Terminal punctuation across the scripts this product actually sees:
# ASCII, Arabic/Persian question mark, and the CJK full stop.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?؟。])\s+")
_QUESTION_END_RE = re.compile(r"[?؟]\s*$")

# Sentence-splitting alone is not enough, and the case that proves it was a
# must-KEEP in this file's own regression set: "I'm really into rock climbing —
# any gyms you'd recommend?" is ONE sentence ending in a question mark, and
# treating it as question-only would have discarded a real stated preference.
#
# So a first-person declarative anywhere in the turn disqualifies it, however
# it is punctuated. The list is verbs of stating, not of asking: "how do I get
# there?" contains "I" and must stay a question.
_FIRST_PERSON_ASSERT_RE = re.compile(
    r"\b(?:i'?m|i\s+am|i'?ve|i\s+have|i\s+had|i\s+live|i\s+lives|i\s+work|"
    r"i\s+like|i\s+love|i\s+hate|i\s+prefer|i\s+need|i\s+want|i\s+use|"
    r"i\s+own|i\s+study|i\s+play|i\s+moved|i\s+started|i\s+got|"
    r"we'?re|we\s+are|my\s+\w+\s+is|my\s+\w+\s+are)\b",
    re.IGNORECASE,
)


def is_question_only(text: Optional[str]) -> bool:
    """True when the user's turn asks and states nothing.

    Deliberately conservative in both directions: a turn with no terminal
    punctuation at all is NOT question-only ("i live in vancouver now" has
    none either), and neither is one carrying a first-person declarative.
    """
    if not text or not text.strip():
        return False
    if _FIRST_PERSON_ASSERT_RE.search(text):
        return False
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    if not parts:
        return False
    return all(_QUESTION_END_RE.search(p) for p in parts)


def inferred_interest_reason(
    content: str, user_message: Optional[str]
) -> Optional[str]:
    """Refuse "the user is interested in X" when X is all they asked about."""
    if not user_message:
        return None
    _quoted, own_voice = _split_quoted(user_message)
    text = own_voice if own_voice.strip() else user_message
    if not is_question_only(text):
        return None
    if _INTEREST_CLAIM_RE.search(content or ""):
        return "inferred_interest"
    return None


def memory_gate_reason(
    content: str,
    *,
    user_message: Optional[str] = None,
    assistant_response: Optional[str] = None,
    explicit_save: bool = False,
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

    # Runs BEFORE the provenance rules: a secret is refused no matter who said
    # it, and unlike the other rules it is not a judgement about relevance.
    reason = sensitive_content_reason(text, explicit_save=explicit_save)
    if reason:
        return reason

    reason = quoted_content_reason(text, user_message)
    if reason:
        return reason

    # An explicit "remember that I'm interested in X" is the user stating it,
    # so this rule — unlike the secret tier — yields to an explicit ask.
    if not explicit_save:
        reason = inferred_interest_reason(text, user_message)
        if reason:
            return reason

    return assistant_echo_reason(text, user_message, assistant_response)


__all__ = [
    "MAX_MEMORY_CHARS",
    "ECHO_ASSISTANT_MIN",
    "ECHO_MARGIN_MIN",
    "QUOTED_CONTAINMENT_MIN",
    "MemoryRejected",
    "SCHEDULABLE_REASONS",
    "SCHEDULED_MIN_TTL_DAYS",
    "assistant_echo_reason",
    "inferred_interest_reason",
    "is_question_only",
    "degenerate_relationship_reason",
    "is_scheduled_commitment",
    "memory_gate_reason",
    "scheduled_floor_ttl",
    "quoted_content_reason",
    "relationship_gate_reason",
    "scaffolding_reason",
    "sensitive_content_reason",
    "transient_horizon_reason",
    "TRANSIENT_HORIZON_DAYS",
    "TRANSIENT_HORIZON_DAYS_PERMANENT",
]
