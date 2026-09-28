"""Per-turn DB attribution — narrowing what a slow phase was WAITING on.

R48. PRODUCTION OBSERVATION (supervisor's platform-log extract for the test
account, 2026-09-20): one sample turn spent 5.023 s inside `[PERF] phase3_save`
— one timer around the save (`agent_runner.py`'s `t_phase3` → the
`[PERF] phase3_save` line). Round 1's map counted six statements there; this
instrument measures FIVE cursor executions on sqlite, because the ORM folds the
two `messages` INSERTs into one `executemany` (see `em` below), and the
statement count on asyncpg has never been observed. Round 1 produced FOUR
mechanisms that all fit that one number and were separated by NOTHING in the
deployed trail:

  * a fresh connection dial. SOURCE DEFAULT, not a verified runtime value: the
    agent engine falls back to NullPool at `config.py: agent_db_pool_size = 0`
    and the repo contains no producer of `AGENT_DB_POOL_SIZE` — but that same
    comment says "Enable per slot with AGENT_DB_POOL_SIZE", so a hand-set
    per-slot override is possible and was NOT checked for the sampled
    container. `conns`/`dials` on the line is what settles it per turn,
  * a wait for a PgBouncer SERVER connection, which in transaction pooling
    happens at the transaction's first query, not at connect (the round-1
    skeptic's correction to a2-F13) — on asyncpg that is INSIDE the first
    `cursor.execute()`, which is what `stmt1_ms` below is for,
  * a slow statement or a row-lock wait (a2-F6: one statement absorbs
    everything and names the contended row),
  * a starved event loop or a CFS-throttled cgroup, where the database was
    never slow at all and no DB fix can help.

None of the four has been refuted, and this module refutes none of them: it is
ONLY an instrument. It changes no behaviour, and with `turn_db_span` off and no
usable canary entry its listeners are never installed at all.

HOW IT ATTRIBUTES. One ContextVar holds the open span; SQLAlchemy engine
events read it. That is the whole design, and it is what makes a GLOBAL
listener safe on a process that runs ~20 background loops on one engine: a
listener fires inside the greenlet SQLAlchemy spawned from the awaiting task,
and greenlet propagates `gr_context`, so `ContextVar.get()` inside the
listener answers the TASK's span — verified against sqlalchemy 2.0.25 with two
concurrent `gather` legs, which interleaved on the wire and were attributed
correctly. No call site is re-timed and every future call site is covered for
free.

WHAT IT EMITS. Two line types and no others. Up to TEN lines per
`AgentRunner.run()` frame: nine `[PERF] db_span` phases (five pre-turn, in
`ws_chat`; four in `agent_runner`) and one `[PERF] turn_host`. "Per frame",
not "per turn": a turn that spawns a sub-agent or a mission runs a SECOND
frame in a context copy, which adds up to four lines of its own (a child does
not save an assistant message, so it has no `save` phase) tagged `nest=1`. An
agentic turn that spawns several adds that many times over — there is no fixed
per-turn ceiling, and any gate that asserts one is asserting something this
instrument does not promise.

WHAT EACH FIELD MEANS (the honest version — read this before concluding
anything from a line):

  ``dial_ms``  `do_connect` → pool `connect`: the CLIENT-SIDE connect, which
               includes DNS, the TCP handshake, TLS if configured and SCRAM
               auth. It is the time to get a CLIENT connection — through
               PgBouncer that is a connection to the POOLER, and it says
               nothing about whether a server connection was available behind
               it (that wait lands in `stmt1_ms`).
  ``conns``    pool checkouts. ``dials`` is how many of them re-dialled; on
               NullPool the two are equal by construction, and that equality
               is itself the pool-shape check `db_pool_checked_out` gives —
               which is also how this line reports whether the deployed pool
               is actually NullPool, rather than assuming the source default.
  ``sql_ms``   exact. `before_cursor_execute` → `after_cursor_execute`.
  ``begin_ms`` the `begin` event to the NEXT wire event on that connection.
               ON THE PRODUCTION DRIVER THIS IS CLIENT-SIDE PYTHON, NOT A
               ROUND TRIP, and an earlier draft of this module said the
               opposite. Verified against the shipped venv (SQLAlchemy 2.0.25
               / asyncpg 0.31.0): `PGDialect_asyncpg.do_begin` IS
               `DefaultDialect.do_begin`, whose body is `pass`, and
               `Connection._begin_impl` fires the `begin` event immediately
               before calling it. asyncpg issues the real `BEGIN` lazily from
               `AsyncAdapt_asyncpg_connection._start_transaction()`, called
               inside `cursor.execute()` — i.e. BETWEEN this module's
               `before_cursor_execute` and `after_cursor_execute` of the first
               statement. So `begin_ms` is the ORM's statement-compilation
               time and nothing else, on asyncpg AND on aiosqlite (same no-op
               `do_begin`), which is why every test engine reports it as ~0.
               Do not read a pooler wait out of it. It stays on the line
               because an unlabelled residue is how a2's own signature table
               ended up with "seconds unattributed" — but it is a residue,
               not a reading.
  ``stmt1_ms`` the first statement executed on a connection THIS span dialled.
               It is a COMPOSITE and it is NOT a pooler timer: it contains the
               BEGIN round trip, any PgBouncer server-assignment wait (in
               transaction pooling the server connection is assigned at the
               first query of the transaction), asyncpg's per-connection
               initialisation and type introspection, the Parse/Bind/Execute of
               the statement itself, and the server's own execution of it — all
               inside one `cursor.execute()`, with nothing on the client able
               to separate them. ``stmt1_ms`` is a SUBSET of ``sql_ms``, never
               an addition to it. `stmt1_ms` ≈ `sql_ms` with `dials≥1` and a
               `slow=` naming that same first statement is the
               fresh-connection signature — it says the cost is in getting a
               usable transaction started, not WHICH of the four things above
               cost it. Zero when the connection was dialled outside the span:
               the dial marks the connection with the SPAN'S OWN id and only a
               matching id counts, so `dials=0` really does imply
               `stmt1_ms=0`, including on a pool that retains connections
               between spans (a bare boolean mark did not — LOCAL repro on
               QueuePool at R48 final review reported `dials=0 … stmt1_ms=125`
               for a span that inherited an unconsumed mark).
  ``nest``     which `AgentRunner.run()` FRAME the line belongs to. 0 is the
               turn the user's message started. A sub-agent or mission run is
               launched with `create_task` from inside that turn, so it is a
               separate `run()` in a context COPY: its spans and its own
               `turn_host` line ride the SAME `cmid_h` (they are part of the
               same user-visible turn) at `nest=1`. Filter on `nest=0` for the
               parent turn alone. Two SIBLINGS at the same depth are not
               distinguished — see the CANNOT list.
  ``commit_ms`` the `commit`/`rollback` event to the NEXT wire event on that
               connection, or to the end of the span. So it is also a
               composite: the COMMIT round trip, the session close and the
               connection's disposal (a Terminate, on NullPool), and any
               Python that ran between the commit event and that next event —
               including time the task spent descheduled. Reset-on-return
               rollback is NOT seen: the pool calls `dialect.do_rollback` on
               the DBAPI connection directly.
  ``stmts``    cursor executions that REACHED `after_cursor_execute`. One that
               raised is timed into `sql_ms` (by `_close_pending`) but not
               counted, so `slow=…@ms` can exceed what `stmts` implies on an
               error path. `em` is how many of them were executemany batches.
  ``loop_lag_max_ms`` the worst event-loop overshoot observed DURING the span.
               Not `loop_health`'s: that sampler runs at 250 ms and
               `snapshot()` answers over a fixed 30 s window shared with every
               background loop and with adjacent turns, so it cannot attribute
               to a 300 ms phase (the round-1 skeptic's objection to a3's
               FIX-E). This one samples at 50 ms, runs ONLY while a span is
               open, and answers "max lag since t".
  ``lag_cov``  1 when the sample ring reaches back past the span's start, 0
               when it does not and `loop_lag_max_ms` is therefore a floor.
               Same rule as `src=` on the host line: an under-reading and a
               true zero must not print alike.

WHAT THIS INSTRUMENT CAN AND CANNOT DISTINGUISH. Read this before writing a
conclusion from a line. It is a CLIENT-SIDE instrument: every bucket is the
interval between two events in this process, so it can say WHERE IN THE
CLIENT'S OWN SEQUENCE the wall time went, and it can never say what the far
end was doing.

It can separate: connection establishment (`dial_ms`) from work on an
established connection; the first statement of a transaction on a fresh
connection (`stmt1_ms`) from later statements; one slow statement (`slow=`,
which names the table) from many ordinary ones (`stmts`, `seq`); a phase whose
wall time is NOT in any DB bucket (`total_ms` ≫ the buckets) from one that is;
and a process that could not run (`loop_lag_max_ms`, `[PERF] turn_host`) from
one that was waiting on I/O.

It CANNOT separate, at all:

  * server-side SQL execution from network transit from a PgBouncer
    server-assignment wait. All three are inside one `cursor.execute()` and
    the client sees one number. Splitting them needs `pg_stat_statements` /
    `pg_stat_activity` / PgBouncer's `SHOW POOLS` on the server side;
  * a lock wait from a genuinely slow query — `slow=update:day_chats@3510`
    names the STATEMENT, and a lock wait is only the leading hypothesis for
    it, never a reading of it;
  * host-level CPU pressure from this container's own cgroup quota; that is
    what the separate `[PERF] turn_host` line's `nr_throttled` vs
    `psi_cpu_some_ms` is for, and `src=none` means neither was readable;
  * which of two connections held concurrently inside one span a bucket
    belongs to. Buckets are SUMMED across connections, so a span that opens
    two sessions (phase 1 does, via `should_use_day_chat_context()`) can have
    two overlapping intervals added as if they were serial. `conns>1` on a
    line means the buckets are an UPPER BOUND on elapsed DB time, and any
    accounting arithmetic over that line double counts;
  * two CONCURRENT SIBLING child runs. `nest=` separates a child run's lines
    from its parent's, and nothing separates one child from another: two
    sub-agents spawned by the same turn both log `nest=1` under the same
    `cmid_h`, so a `phase=phase1` line at `nest=1` cannot be assigned to one
    of them. Sequential children are equally unlabelled. What `nest=` buys is
    that a child's work is no longer counted as the parent's, which is what it
    was before (blocking finding B1);
  * WHERE a nested run's wall time sits inside the parent's. A `nest=1` line
    is timed in the child's own frame; the parent's `total_ms` for a phase
    does not contain it, and the two are not intervals of one clock on the
    line. Joining them needs the `[TURN_WATERFALL]`/job trail, not this one.

And one failure mode inflates EVERY bucket at once: a starved event loop. Each
bucket is closed by a callback that has to be scheduled, so a loop that cannot
run stretches `dial_ms`, `sql_ms`, `stmt1_ms`, `commit_ms` and `total_ms`
together and the shape looks exactly like a uniformly slow database.
`loop_lag_max_ms` (with `lag_cov=1`) is the discriminator — NOT the buckets.

DECISION TABLE — pattern → what it SUPPORTS → what it does NOT exclude:

  `dial_ms` ≫ rest
     supports: connect cost — TCP/TLS/SCRAM, or a pooler refusing clients.
     does not exclude: DNS, a saturated host slowing the handshake.

  `stmt1_ms` ≈ `sql_ms`, `dials≥1`, `slow=` names statement #1
     supports: the cost is in starting a usable transaction on a fresh
       connection.
     does not exclude: any of {PgBouncer server wait, asyncpg type
       introspection, a genuinely slow first query, a lock taken by it}. This
       pattern is the reason to go look at the server; it is not the answer.

  a big `slow=` on a LATER statement, `stmt1_ms` small
     supports: one statement dominates, and the line names its table.
     does not exclude: lock wait vs slow plan vs a large result set. Needs
       `pg_stat_activity` / `EXPLAIN` to go further.

  `loop_lag_max_ms` in the hundreds, `lag_cov=1`
     supports: the process could not run; DB fixes cannot help this turn.
     does not exclude: a slow database AS WELL — lag inflates the buckets, so
       a real DB wait underneath is neither shown nor ruled out.

  `total_ms` ≫ sum of buckets, `conns≤1`, `loop_lag_max_ms` small
     supports: wall time inside the span that is not connect, not a statement,
       not a commit — e.g. Python between statements, or a wait this module
       has no event for.
     does not exclude: an event this module does not subscribe to. It is a
       residue, not a fifth mechanism.

  buckets sum to ≥ `total_ms`, `conns>1`
     supports: nothing. Overlapping connections were summed; see above.

  a second `phase=…` under one `cmid_h`, at `nest≥1`
     supports: this turn spawned a child run (a sub-agent, a mission), and
       that line is the CHILD's DB work.
     does not exclude: several children. `nest` is a depth, not an identity —
       two siblings both print `nest=1`. It also does not tell you whether the
       parent was waiting on the child or had already answered.

  `begin_ms` non-zero
     supports: ORM statement-compilation time on the drivers this repo ships.
     does not exclude: nothing — it contains no round trip at all here (above),
       so no conclusion may rest on it until a driver whose `do_begin` is not
       a no-op is deployed.

PRIVACY. Statement identity is a `verb:table` token built from two fixed
whitelists — never SQL text, never parameters. A `messages` INSERT's
parameters ARE the user's message (which is why `database.py` sets
`hide_parameters=True`), so `_tok` is a structural guard, not a convenience:
nothing it can return is derived from free text. Correlation is `cmid_h`, the
FNV-1a hash the three-hop turn trace already uses, never a raw id.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import deque
from contextvars import ContextVar
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ── gating ──────────────────────────────────────────────────────────

def _flag(name: str, default: Any = False) -> Any:
    try:
        from app.config import settings
        return getattr(settings, name, default)
    except Exception:  # noqa: BLE001 — a diagnostic may never fail a boot
        return default


_SETTING = "TURN_DB_SPAN_CANARY_USER_IDS"

#: A user id here is the platform `users.id` — a uuid4 string, 36 characters
#: hyphenated (32 bare). An entry shorter than 32 characters cannot be one.
#:
#: This bound exists because of a real operational defect, not a hypothetical:
#: this patch's OWN rollout instructions told an operator to set
#: `TURN_DB_SPAN_CANARY_USER_IDS` to the 8-character prefix that names the
#: CONTAINER (`toup-agent-xxxxxxxx`) and is what appears in every log line.
#: Matching here is EXACT, so that would have armed the process, opened no
#: span, printed no line, and read on every acceptance gate as a healthy,
#: quiet control. A silent no-op is the one outcome an instrument must never
#: have, so a short entry is now rejected LOUDLY and never enables anything.
_MIN_PLAUSIBLE_ID_LEN = 32

#: Entries already reported. Log-deduplication ONLY: keyed by the configured
#: STRING, never by tenant or by a decision, and it can only suppress a
#: repeated ERROR — never change whether a span opens. That is why it is not
#: the kind of process-level memo the repo bans (a memo that answers for the
#: previous tenant after `/admin/bind`): the gate itself is still re-derived
#: from settings on every call.
_reported: set = set()


#: At most this many characters of a configured entry are ever written to the
#: log. Eight, because that is exactly what a container name (`toup-agent-
#: xxxxxxxx`) and every `[REALTIME]`-style line in this repo already carry, so
#: an operator recognises what they typed — and because the FIRST EIGHT
#: characters of a uuid are the one part that is already in the log stream
#: anyway. An earlier draft truncated only at `_MIN_PLAUSIBLE_ID_LEN`, which
#: meant a 31-character entry — 86 % of a hyphenated uuid — printed WHOLE from
#: the code whose whole purpose is to complain about pasted ids. Measured
#: LOCAL at R48 final review: 31 → printed whole, 32 → truncated.
_SHOW_CHARS = 8


def _show(entry: str) -> str:
    """How a rejected entry is rendered in the ERROR line.

    NEVER more than the first `_SHOW_CHARS` characters, at any length. An
    entry of `_SHOW_CHARS` or fewer prints whole (there is nothing to hide —
    that is the container-name prefix the operator most likely pasted, and
    they have to recognise it); anything longer prints its first eight
    characters and its LENGTH, which is what distinguishes "they pasted a
    prefix" from "they pasted something else entirely".
    """
    try:
        if len(entry) <= _SHOW_CHARS:
            return repr(entry)
        return "%r…(%d chars)" % (entry[:_SHOW_CHARS], len(entry))
    except Exception:  # noqa: BLE001
        return "<unprintable>"


def _report_once(key: str, message: str) -> None:
    try:
        if key in _reported:
            return
        _reported.add(key)
        logger.error(message)
    except Exception:  # noqa: BLE001 — a diagnostic may never fail a turn
        pass


def _canary_entries() -> frozenset:
    """The list EXACTLY as configured, valid or not."""
    raw = _flag("turn_db_span_canary_user_ids", "") or ""
    return frozenset(u.strip() for u in str(raw).split(",") if u.strip())


def _usable_canary_ids(entries: Optional[frozenset] = None) -> frozenset:
    """The entries that could possibly equal a user id, reporting the rest.

    One ERROR per process per bad entry. Never raises, and never widens the
    match: a rejected entry is dropped, it is not turned into a prefix rule.
    """
    ids = _canary_entries() if entries is None else entries
    if not ids:
        return ids
    usable = frozenset(i for i in ids if len(i) >= _MIN_PLAUSIBLE_ID_LEN)
    for bad in sorted(ids - usable):
        _report_once(
            f"short:{bad}",
            "[db_span] %s entry %s is %d characters; a user id is a %d-character "
            "uuid. Matching is EXACT — this entry is IGNORED and enables nothing. "
            "If this is the 8-character prefix from a container name or a log "
            "line, it is not a user id: resolve the full uuid from the platform "
            "users table." % (_SETTING, _show(bad), len(bad),
                              _MIN_PLAUSIBLE_ID_LEN + 4),
        )
    return usable


def armed() -> bool:
    """Whether the engine listeners should be installed in this process.

    Read ONCE per engine build. The fleet-wide flag is off and the canary list
    is empty by default, so on an ordinary container nothing is registered and
    the instrument is not merely cheap but absent.

    WHEN EVERY ENTRY IS INVALID this returns False, and says so once at ERROR.
    The alternative — arm anyway — was rejected: `span_enabled` would refuse
    every one of those entries too, so the listeners could not open a single
    span, and the process would pay eight engine callbacks on every statement
    of every background loop to measure nothing. It also fails in the safe
    direction: the worst case is LESS instrumentation than the operator asked
    for, never a span opened for a user they did not name. A list with ONE
    usable entry still arms — the bad entries are reported and dropped.
    """
    if bool(_flag("turn_db_span", False)):
        return True
    entries = _canary_entries()
    if not entries:
        return False
    if _usable_canary_ids(entries):
        return True
    _report_once(
        "all-rejected",
        "[db_span] every %s entry was rejected (see above); the instrument is "
        "NOT armed and will emit nothing. Set the FULL user uuid." % _SETTING,
    )
    return False


def span_enabled(user_id: Optional[str]) -> bool:
    """Global-or-canary, the same shape as `stable_prefix_enabled` /
    `channel_envelope_enabled` — agent flags are otherwise fleet-wide, and a
    canary list is the only way to prove this on one tenant first.

    Matching is EXACT. A turn whose `user_id` merely STARTS WITH an entry is
    the precise signature of a pasted log prefix, so it is reported once per
    process per entry at ERROR — and still does NOT enable the span. Reporting
    a prefix is not the same as honouring one: a prefix rule would let one
    mistyped entry instrument an arbitrary set of tenants.
    """
    if bool(_flag("turn_db_span", False)):
        return True
    entries = _canary_entries()
    if not entries or not user_id:
        return False
    if user_id in _usable_canary_ids(entries):
        return True
    for entry in sorted(entries):
        if entry != user_id and user_id.startswith(entry):
            _report_once(
                f"prefix:{entry}",
                "[db_span] %s entry %s is a PREFIX of a real user id, not the "
                "id. Matching is EXACT, so no span was opened and no "
                "measurement will be produced. Set the full uuid (resolve it "
                "from the platform users table; the 8-character prefix in a "
                "log line or a container name never matches)."
                % (_SETTING, _show(entry)),
            )
    return False


# ── statement identity: `verb:table`, from whitelists only ──────────

#: Every table the turn path touches, per the round-1 round-trip map. A name
#: outside this set renders as `other` — the whitelist is the privacy
#: guarantee, so it is a fixed literal and never derived from the statement.
_TABLES = frozenset({
    "users", "conversations", "messages", "day_chats", "agent_configs",
    "identities", "memory_files", "migration_status", "processed_messages",
    "context_budget_logs", "memories", "documents", "entities", "jobs",
    "automations", "automation_runs", "build_jobs", "attachments",
})

_VERBS = ("select", "insert", "update", "delete", "savepoint", "release",
          "rollback", "begin", "commit", "set", "show", "with")

_VERB_RE = re.compile(r"^\s*\(?\s*([a-z]+)", re.I)
_FROM_RE = re.compile(r"\bfrom\s+\"?([a-z_][a-z0-9_]*)\"?", re.I)
_INTO_RE = re.compile(r"\binto\s+\"?([a-z_][a-z0-9_]*)\"?", re.I)
_UPDATE_RE = re.compile(r"\bupdate\s+\"?([a-z_][a-z0-9_]*)\"?", re.I)

#: Only the head of a statement is ever scanned — a 200 KB INSERT must not
#: become a 200 KB regex walk on the turn path. 4 KB rather than a few dozen
#: bytes because an ORM `select(Conversation)` renders every mapped column
#: before its `FROM`: at 160 chars every phase-1 SELECT tokenised as
#: `select:-` and the instrument reported a shape it could not see. The
#: whitelist is what keeps this safe: even a match found inside a parameter
#: can only render as a whitelisted table name or `other`.
_SCAN_CHARS = 4000

#: How many statement tokens ride the emitted line. Twelve covers the save
#: (five cursor executions on sqlite — see `em`) and each pre-turn block with
#: room to spare; phase 1 is longer and is truncated with a count, which is
#: the honest rendering of "more than this fits".
_MAX_SEQ = 12


def _tok(statement: Any) -> str:
    """`verb:table` for one statement. NEVER returns SQL text or parameters.

    Both halves are looked up in fixed whitelists, so the return value is
    drawn from a finite set that contains no user content by construction.
    Anything unrecognised is `other`.
    """
    try:
        head = str(statement)[:_SCAN_CHARS]
    except Exception:  # noqa: BLE001
        return "other:-"
    m = _VERB_RE.match(head)
    verb = (m.group(1).lower() if m else "")
    if verb not in _VERBS:
        return "other:-"
    if verb == "insert":
        t = _INTO_RE.search(head)
    elif verb == "update":
        t = _UPDATE_RE.search(head)
    elif verb in ("select", "delete", "with"):
        t = _FROM_RE.search(head)
    else:
        return f"{verb}:-"
    if t is None:
        return f"{verb}:-"
    name = t.group(1).lower()
    return f"{verb}:{name if name in _TABLES else 'other'}"


# ── the span ────────────────────────────────────────────────────────

#: Monotonic span id. It exists ONLY so the `_db_span_fresh` mark a dial
#: leaves on a connection can name the span that left it: a span that dials a
#: connection and never executes a statement on it (a session opened and
#: closed without a query) leaves that mark behind, and on a connection-
#: RETAINING pool the NEXT span to check that connection out would have
#: consumed it and charged its own first statement to `stmt1_ms` — with
#: `dials=0` on the same line, which is the one combination the field's
#: docstring promises cannot happen. Reproduced LOCAL on QueuePool at R48
#: final review (`dials=0 … stmt1_ms=125`). A counter rather than `id(span)`
#: because an id is reused once the span is freed, which is precisely when
#: this comparison is asked.
_SPAN_SEQ = 0


class _Span:
    __slots__ = (
        "phase", "t0", "conns", "dials", "dial_ms", "begin_ms", "commit_ms",
        "stmts", "em", "sql_ms", "slow_tok", "slow_ms", "closed", "pending",
        "toks", "stmt1_ms", "sid",
    )

    def __init__(self, phase: str) -> None:
        global _SPAN_SEQ
        _SPAN_SEQ += 1
        self.sid = _SPAN_SEQ
        self.phase = phase
        self.t0 = time.perf_counter()
        self.conns = 0
        self.dials = 0
        self.dial_ms = 0.0
        self.begin_ms = 0.0
        # The first statement on a connection this span dialled. A SUBSET of
        # `sql_ms`, never added to it — on asyncpg the BEGIN, the pooler's
        # server assignment and the driver's type introspection all happen
        # inside that one `cursor.execute()`, which is why `begin_ms` cannot
        # carry them and this field exists (R48 review, blocking finding B-1).
        self.stmt1_ms = 0.0
        self.commit_ms = 0.0
        self.stmts = 0
        # How many of `stmts` were executemany batches. SQLAlchemy's unit of
        # work folds this turn's TWO `messages` INSERTs into ONE cursor
        # execution, so the save is 5 cursor executions, not the 6 the round-1
        # map counted — and on asyncpg a 2-row executemany is one Parse plus
        # two Bind/Execute, so the round-trip arithmetic both a1 and a2 built
        # on "6 statements" has to be re-derived from this field.
        self.em = 0
        self.sql_ms = 0.0
        self.slow_tok = ""
        self.slow_ms = 0.0
        # The statement SHAPE of the phase, in order, bounded. Tokens only —
        # a finite whitelist, so this carries no user content — and it rides
        # the line rather than a test-only accessor because "the save is N
        # statements in this order" is an assertion worth being able to make
        # against PRODUCTION, not only against a fixture — and the local
        # figure (five cursor executions on sqlite, not the six round 1's map
        # counted) is exactly the kind of number a fixture cannot settle.
        self.toks: list = []
        self.closed = False
        # id(Connection) -> (the Connection, kind, t0). `kind` is "begin" /
        # "commit" / "rollback" / a statement token; see the module docstring
        # for why these are closed by the NEXT event rather than by their own.
        #
        # The Connection OBJECT is held, not just its id, and that is the
        # whole reason the tuple has three fields: a `("commit", t)` entry is
        # closed by the next wire event on that connection or by span exit, so
        # a span that opens two sessions (phase 1 opens at least two) could
        # otherwise have session B's Connection land at the ADDRESS session
        # A's freed Connection had, and charge the Python time between A's
        # commit and B's first statement to `commit_ms`. A live reference
        # makes that address unreusable for the span's lifetime.
        self.pending: Dict[int, Tuple[Any, str, float]] = {}


_SPAN: ContextVar[Optional[_Span]] = ContextVar("toup_db_span", default=None)


def _cur() -> Optional[_Span]:
    s = _SPAN.get()
    if s is None or s.closed:
        return None
    return s


def _close_pending(s: _Span, key: int, now: float) -> None:
    got = s.pending.pop(key, None)
    if got is None:
        return
    _conn, kind, t0 = got
    ms = (now - t0) * 1000.0
    if kind == "begin":
        s.begin_ms += ms
    elif kind in ("commit", "rollback"):
        s.commit_ms += ms
    else:  # a statement that never saw its `after` event (an error path)
        s.sql_ms += ms


# ── event-loop lag, sampled only while a span is open ───────────────

_LAG_INTERVAL_S = 0.05
#: ~6.8 minutes of history at 50 ms. It was 1024 samples (~51 s), argued as
#: sufficient from the single worst record on file (`phase3_save: 34505ms`,
#: 2026-09-15) — but an agentic turn can hold a span open for minutes, and a
#: ring that has rolled over reports a MAXIMUM it did not observe. The deque
#: only ever fills while a span is open on an armed process, so the ~500 KB
#: ceiling is paid by the canary and by nobody else. `lag_cov` on the line
#: reports the case where even this was not enough, rather than under-reading
#: in silence.
_LAG: Deque[Tuple[float, float]] = deque(maxlen=8192)
_lag_task: Optional["asyncio.Task[None]"] = None
#: The loop `_lag_task` belongs to. A task from ANOTHER loop reads as
#: done/pending from this one's point of view but cannot be AWAITED here, and
#: its samples describe a different process timeline — so the sampler is
#: recreated on a mismatch rather than assumed live. It IS cancelled first,
#: through that loop's own `call_soon_threadsafe`; dropping the reference
#: instead leaks a 50 ms wakeup for that loop's life and, worse, keeps it
#: appending into the shared `_LAG` deque, diluting this loop's
#: `loop_lag_max_ms` with another timeline's. Unreachable in the single-loop
#: agent process, and NOT covered by a test — see NOTES §12.12.
_lag_loop: Optional[Any] = None
_lag_refs = 0


async def _lag_sampler() -> None:
    while True:
        before = time.perf_counter()
        await asyncio.sleep(_LAG_INTERVAL_S)
        now = time.perf_counter()
        # Overshoot beyond the sleep we asked for IS the block.
        _LAG.append((now, max(0.0, (now - before - _LAG_INTERVAL_S) * 1000.0)))


def _lag_acquire() -> None:
    global _lag_task, _lag_refs, _lag_loop
    _lag_refs += 1
    try:
        loop = asyncio.get_running_loop()
    except Exception:  # noqa: BLE001 — no loop, or loop closing
        return
    if _lag_task is not None and not _lag_task.done() and _lag_loop is loop:
        return
    if _lag_task is not None and _lag_loop is not loop:
        # A live sampler on ANOTHER loop. It cannot be awaited from here, but
        # it CAN be cancelled from here (`Task.cancel` is documented as the
        # one thread-unsafe-but-loop-agnostic call the other loop will honour
        # on its next cycle), and dropping the reference without cancelling
        # leaks a 50 ms wakeup for the life of that loop — appending its
        # samples into the SAME `_LAG` deque, so a second loop's timeline
        # silently dilutes this one's `loop_lag_max_ms`.
        try:
            _lag_loop.call_soon_threadsafe(_lag_task.cancel)
        except Exception:  # noqa: BLE001 — loop closed, or not running
            try:
                _lag_task.cancel()
            except Exception:  # noqa: BLE001
                pass
    try:
        _lag_task = loop.create_task(_lag_sampler(), name="db_span.lag")
        _lag_loop = loop
    except Exception:  # noqa: BLE001
        _lag_task = None
        _lag_loop = None


def _lag_release() -> None:
    global _lag_task, _lag_refs, _lag_loop
    _lag_refs = max(0, _lag_refs - 1)
    if _lag_refs or _lag_task is None:
        return
    try:
        _lag_task.cancel()
    except Exception:  # noqa: BLE001
        pass
    _lag_task = None
    _lag_loop = None


def _lag_max_since(t0: float) -> float:
    best = 0.0
    for ts, lag in _LAG:
        if ts >= t0 and lag > best:
            best = lag
    return best


def _lag_covers(t0: float) -> bool:
    """Whether the ring still holds a sample from before `t0`.

    False means it has rolled over inside this span and `_lag_max_since` is a
    floor, not a maximum. An empty ring is `True`: nothing was dropped, there
    was simply nothing to sample yet (a span shorter than one interval)."""
    try:
        return (not _LAG) or _LAG[0][0] <= t0
    except Exception:  # noqa: BLE001
        return False


# ── listeners ───────────────────────────────────────────────────────

def _on_do_connect(dialect, conn_rec, cargs, cparams):  # noqa: ANN001
    # MUST return None: a `do_connect` listener that returns a value REPLACES
    # the connection SQLAlchemy would have made.
    s = _cur()
    if s is None:
        return None
    try:
        conn_rec.info["_db_span_dial_t0"] = time.perf_counter()
    except Exception:  # noqa: BLE001
        pass
    return None


def _on_connect(dbapi_connection, connection_record) -> None:  # noqa: ANN001
    s = _cur()
    if s is None:
        return
    s.dials += 1
    try:
        t0 = connection_record.info.pop("_db_span_dial_t0", None)
        # Mark the connection so the FIRST statement on it can be timed
        # separately (`stmt1_ms`). `connection_record.info` is the same dict
        # the cursor listeners reach as `conn.connection.info`, verified
        # against sqlalchemy 2.0.25. The mark carries THIS span's id, not a
        # bare True: a span that dials and never executes leaves the mark
        # behind, and on a connection-retaining pool the next span to check
        # that connection out would otherwise consume it and report
        # `dials=0 … stmt1_ms>0` (LOCAL repro, R48 final review).
        connection_record.info["_db_span_fresh"] = s.sid
    except Exception:  # noqa: BLE001
        t0 = None
    if t0 is not None:
        s.dial_ms += (time.perf_counter() - t0) * 1000.0


def _on_checkout(dbapi_connection, connection_record, connection_proxy) -> None:  # noqa: ANN001
    s = _cur()
    if s is not None:
        s.conns += 1


def _on_begin(conn) -> None:  # noqa: ANN001
    s = _cur()
    if s is None:
        return
    now = time.perf_counter()
    _close_pending(s, id(conn), now)
    s.pending[id(conn)] = (conn, "begin", now)


def _on_commit(conn) -> None:  # noqa: ANN001
    s = _cur()
    if s is None:
        return
    now = time.perf_counter()
    _close_pending(s, id(conn), now)
    s.pending[id(conn)] = (conn, "commit", now)


def _on_rollback(conn) -> None:  # noqa: ANN001
    s = _cur()
    if s is None:
        return
    now = time.perf_counter()
    _close_pending(s, id(conn), now)
    s.pending[id(conn)] = (conn, "rollback", now)


def _on_before_cursor(conn, cursor, statement, parameters, context,  # noqa: ANN001
                      executemany) -> None:
    s = _cur()
    if s is None:
        return
    now = time.perf_counter()
    _close_pending(s, id(conn), now)
    s.pending[id(conn)] = (conn, _tok(statement), now)


def _on_after_cursor(conn, cursor, statement, parameters, context,  # noqa: ANN001
                     executemany) -> None:
    s = _cur()
    if s is None:
        return
    got = s.pending.pop(id(conn), None)
    if got is None:
        return
    _conn, tok, t0 = got
    ms = (time.perf_counter() - t0) * 1000.0
    s.stmts += 1
    if executemany:
        s.em += 1
    s.sql_ms += ms
    # The first statement on a connection THIS span dialled carries the
    # pooler's server assignment, the driver's type introspection and (on
    # asyncpg) the BEGIN itself. Separated out because `begin_ms` structurally
    # cannot hold any of them — see the module docstring.
    try:
        # Popped unconditionally — a mark left by an EARLIER span is stale and
        # must be cleared, not inherited — but only counted when it is this
        # span's own. That comparison is what makes `dials=0 ⇒ stmt1_ms=0`
        # true on a pool that retains connections.
        if conn.connection.info.pop("_db_span_fresh", None) == s.sid:
            s.stmt1_ms += ms
    except Exception:  # noqa: BLE001 — a closed or detached connection
        pass
    if len(s.toks) < _MAX_SEQ:
        s.toks.append(tok)
    if ms > s.slow_ms:
        s.slow_ms = ms
        s.slow_tok = tok


_INSTALLED_MARK = "_toup_db_span_installed"


def install(engine) -> bool:  # noqa: ANN001
    """Register the listeners on an AsyncEngine (or a sync Engine).

    Idempotent per engine, and unconditional — the CALLER decides whether the
    process is armed (`database._build_engine` asks `armed()`), so a test can
    install on its own engine without touching process settings.
    """
    try:
        from sqlalchemy import event

        target = getattr(engine, "sync_engine", engine)
        if getattr(target, _INSTALLED_MARK, False):
            return False
        setattr(target, _INSTALLED_MARK, True)
        event.listen(target, "do_connect", _on_do_connect)
        event.listen(target, "connect", _on_connect)
        event.listen(target, "checkout", _on_checkout)
        event.listen(target, "begin", _on_begin)
        event.listen(target, "commit", _on_commit)
        event.listen(target, "rollback", _on_rollback)
        event.listen(target, "before_cursor_execute", _on_before_cursor)
        event.listen(target, "after_cursor_execute", _on_after_cursor)
        return True
    except Exception as e:  # noqa: BLE001 — an instrument may never fail a boot
        logger.debug("[db_span] install failed: %r", e)
        return False


def is_installed(engine) -> bool:  # noqa: ANN001
    return bool(getattr(getattr(engine, "sync_engine", engine),
                        _INSTALLED_MARK, False))


# ── the turn: identity, gating and host counters ────────────────────

class _Turn:
    __slots__ = ("cmid_h", "channel", "enabled", "host0", "host_emitted",
                 "nest", "run_depth")

    def __init__(self, cmid_h: str, channel: str, enabled: bool) -> None:
        self.cmid_h = cmid_h
        self.channel = channel
        self.enabled = enabled
        self.host0: Dict[str, int] = {}
        self.host_emitted = False
        #: What `nest=` prints: 0 for the turn the user's message started,
        #: 1 for a run spawned from inside it (a sub-agent, a mission), 2 for
        #: a run that one spawned, and so on.
        self.nest = 0
        #: Which `AgentRunner.run()` FRAME owns this turn — i.e. the
        #: `_RUN_NEST` value at which `emit_turn_host()` / `end_turn()` may
        #: act on it. `None` means "armed outside `run()`" (the WS handler),
        #: and the first frame to enter adopts it. See `enter_run`.
        self.run_depth: Optional[int] = None


_TURN: ContextVar[Optional[_Turn]] = ContextVar("toup_db_span_turn", default=None)

#: How many `AgentRunner.run()` frames enclose this context. NOT state on the
#: `_Turn`: a `_Turn` is shared BY REFERENCE with every context copied from
#: this one (`asyncio.create_task`, `background_tasks.spawn`), whereas a
#: ContextVar write is private to the context that makes it — which is exactly
#: the distinction a parent/child pair needs and the reason blocking finding
#: B1 existed. Zero in the WS handler, 1 inside the turn's own `run()`, 2
#: inside a sub-agent run spawned from a tool during that turn.
_RUN_NEST: ContextVar[int] = ContextVar("toup_db_span_run_nest", default=0)

#: `channel` is RAW CLIENT INPUT — `msg.get("channel")` straight off the WS
#: frame, armed two lines above the handler's own "validate client_tz before
#: ANY of it is believed" block. It is rendered as the last field of both
#: lines, so a value containing a newline forges a `[PERF] db_span` line in
#: the very stream this investigation reads, and a value containing a space
#: silently breaks the key=value grep contract. Whitelist, don't escape:
#: anything outside this class is deleted.
_CH_SAFE_RE = re.compile(r"[^a-z0-9_.-]")


def _safe_channel(channel: Any) -> str:
    try:
        cleaned = _CH_SAFE_RE.sub("", str(channel or "").lower())[:16]
    except Exception:  # noqa: BLE001
        return "-"
    return cleaned or "-"


def enter_run() -> Any:
    """Mark that an `AgentRunner.run()` frame starts HERE. Returns a token.

    Two jobs, both of them consequences of blocking finding B1 (R48 final
    review, reproduced LOCAL):

    1. **Adoption.** A turn armed by the WS handler is armed at nest 0, and
       the `run()` task that consumes it is a CONTEXT COPY of the handler's.
       The first frame to enter claims it, so `emit_turn_host()` and
       `end_turn()` can tell "my turn" from "an enclosing frame's turn".
    2. **Depth.** A sub-agent or mission is launched with `create_task` from
       INSIDE a live turn (`subagent_orchestrator.py`'s
       `_spawn_bg(_run_child(...))`, `subagent.py`'s
       `create_task(self._execute(...))`), so the child's context copy holds
       the parent's `_Turn` OBJECT. Before this, the child's `run()` finally
       emitted `[PERF] turn_host` under the PARENT's `cmid_h` and channel,
       set `host_emitted` on the shared object, and the parent — still armed
       — emitted nothing at all. One host line existed, so every acceptance
       gate stayed green on a reading that described a different span of
       time than it claimed.

    The token MUST be handed back to `exit_run` in the same `finally`, or a
    caller that awaits `run()` repeatedly on ONE context (`ws_router`'s
    `while True`, several `_think`s per voice call, the heartbeat's per-user
    loop, cron) would see turn 2 as a child of turn 1.
    """
    try:
        depth = _RUN_NEST.get() + 1
        nest_tok = _RUN_NEST.set(depth)
        turn_tok = None
        t = _TURN.get()
        if t is not None and t.run_depth is None:
            t.run_depth = depth
        if depth > 1:
            # A NESTED frame makes the enclosing frame's `_TURN` restorable.
            # Launched with `create_task` it writes to a context copy and this
            # is redundant; awaited DIRECTLY from inside `run()` it does not,
            # and `begin_turn`'s mint would replace the parent's turn in the
            # parent's own context — the parent would then emit the CHILD's
            # line. No path read at 4f0e9fe1 awaits `run()` from inside
            # `run()` (both child paths are `create_task`; telegram, cron,
            # heartbeat, voice and the routines runner all await it at top
            # level), so this is the frame being hermetic whatever a future
            # child does, not a fix for a live bug.
            turn_tok = _TURN.set(t)
        return (nest_tok, turn_tok)
    except Exception:  # noqa: BLE001 — telemetry never costs a turn
        return None


def exit_run(token: Any) -> None:
    """Undo `enter_run`. Must run AFTER `emit_turn_host()`/`end_turn()`, which
    both read `_RUN_NEST` to decide whether this frame owns the turn."""
    try:
        if not token:
            return
        nest_tok, turn_tok = token
        if turn_tok is not None:
            _TURN.reset(turn_tok)
        _RUN_NEST.reset(nest_tok)
    except Exception:  # noqa: BLE001 — a token minted in another context
        try:
            _RUN_NEST.set(0)
        except Exception:  # noqa: BLE001
            pass


def _own_turn() -> Optional[_Turn]:
    """The turn this `run()` frame owns, or None.

    None when there is no turn, or when the visible one belongs to an
    ENCLOSING frame — the B1 case: a child run whose `begin_turn` never got
    to mint its own turn (it raised, or the child never called it) must not
    consume the parent's host line.
    """
    t = _TURN.get()
    if t is None:
        return None
    if t.run_depth is not None and t.run_depth != _RUN_NEST.get():
        return None
    return t


def begin_turn(*, user_id: Optional[str], client_msg_id: Optional[str],
               channel: Optional[str], replace: bool = False) -> None:
    """Arm this turn's spans and stamp its identity.

    Called from the WS handler BEFORE `create_task(runner.run(...))` — a task
    copies the context at creation, so the runner's spans inherit this — and
    again from `_run_inner` for turns that never touch a socket (cron,
    Telegram, a routine). That second call is a no-op while a turn is armed,
    so the runner never overwrites the WS handler's identity.

    `replace=True` is for the WS handler alone: its context is shared by every
    later iteration of the receive loop, so the second message on a socket
    MUST mint a new turn rather than inherit the first one's `cmid_h` and its
    already-taken host counters.

    A NESTED run mints its own turn without `replace`. A sub-agent's `run()`
    executes in a context COPY that holds the parent's `_Turn` by reference,
    so "a turn is already armed, keep it" made the child's spans, its channel
    and its host line indistinguishable from the parent's (blocking finding
    B1). A frame deeper than the one that owns the visible turn therefore gets
    a turn of its own: its OWN channel (`subagent`, not the parent's `web`),
    its OWN host-counter baseline, its own `host_emitted` — and it INHERITS
    the parent's `cmid_h`, because a child's DB work is part of the same
    user-visible turn and `cmid_h` is what joins them. `nest=` on the line is
    what tells the two apart. Concurrent SIBLINGS at the same depth are NOT
    separable; that is in the module docstring's CANNOT list.

    Deliberately NOT memoised on anything process-level: `enabled` is a
    function of `user_id`, and `/admin/bind` swaps the tenant under a live
    process, so a memo would answer for the previous tenant. It is re-derived
    per turn (a set build over a short env string).
    """
    try:
        cur = _TURN.get()
        depth = _RUN_NEST.get()
        nested = (cur is not None and cur.run_depth is not None
                  and depth > cur.run_depth)
        if cur is not None and not replace and not nested:
            return
        from app.services.cmid import cmid_hash

        enabled = span_enabled(user_id)
        t = _Turn(
            # `str(...)` first: direct AgentRunner callers can pass malformed
            # client input. A raw slice would raise, be swallowed, and
            # silently leave the turn unarmed. The WS handler first normalizes
            # malformed ids to None so its pre-turn hash matches the runner.
            cur.cmid_h if nested and cur is not None
            else cmid_hash(str(client_msg_id or "")[:100]),
            _safe_channel(channel),
            enabled,
        )
        # 0 in the WS handler (adopted by the first `run()` frame, which sets
        # `run_depth`); otherwise this frame's depth, and `nest=` counts the
        # frames ABOVE the turn's own.
        t.run_depth = depth or None
        t.nest = depth - 1 if depth else 0
        if enabled:
            t.host0 = read_host_counters()
        _TURN.set(t)
    except Exception:  # noqa: BLE001 — telemetry never costs a turn
        pass


def end_turn() -> None:
    """Drop this turn's arming. Called from `AgentRunner.run()`'s `finally`.

    Without it, `begin_turn`'s "a turn is already armed, keep the handler's
    identity" rule silently becomes "the FIRST turn in this context owns every
    later one". `replace=True` covers only the WS receive loop; every other
    caller awaits `run()` directly in a loop on ONE task context —
    `ws_router.py`'s `while True:`, `ws_realtime`'s several `_think`s per voice
    call, `heartbeat_service`'s `for user_id, chat_id in user_chats:`,
    `cron_service` — so turns 2..N inherited turn 1's `cmid_h` and, because
    `host_emitted` was already True, emitted no `[PERF] turn_host` at all.
    Worse across users: a heartbeat cycle that visited a non-canary user first
    left the CANARY's own turn unarmed under the other user's `cmid_h`.
    Reproduced against the shipped code at R48 review (blocking finding B-2);
    `test_two_sequential_runs_are_two_turns` is the guard.

    Safe on the WS path: `run()` executes in a `create_task` context COPY, so
    clearing the var there cannot disturb the handler's own context — and the
    handler re-arms with `replace=True` on its next message regardless.

    Unguarded on purpose. A `_own_turn()` check here would be decoration: in
    a `create_task` child this writes to a context copy nobody else reads, and
    in a directly-awaited nested frame `exit_run`'s restore undoes it either
    way — so no mutation of such a guard could be caught by any test, and a
    guard that cannot fail is not a guard."""
    try:
        _TURN.set(None)
    except Exception:  # noqa: BLE001
        pass


def turn_enabled() -> bool:
    t = _TURN.get()
    return bool(t is not None and t.enabled)


def turn_cmid_h() -> str:
    t = _TURN.get()
    return t.cmid_h if t is not None else "00000000"


# ── host pressure, read from INSIDE the container ───────────────────
#
# The "0.5 vs 1.0 CPU, throttled or not" question (round 1 found THREE
# different limits for the same container name depending on which of four
# creation paths made it) is answerable without touching the host: a cgroup
# publishes its own throttling inside the namespace. PSI adds the other half
# — throttling says "the quota ran out", pressure says "runnable tasks
# waited", which is what a noisy neighbour on a shared VPS looks like.

_CGROUP_CPU_STAT = (
    "/sys/fs/cgroup/cpu.stat",            # cgroup v2 (unified)
    "/sys/fs/cgroup/cpu/cpu.stat",        # cgroup v1
    "/sys/fs/cgroup/cpu,cpuacct/cpu.stat",
)
_PRESSURE_FILES = {
    "cpu": ("/sys/fs/cgroup/cpu.pressure", "/proc/pressure/cpu"),
    "mem": ("/sys/fs/cgroup/memory.pressure", "/proc/pressure/memory"),
    "io": ("/sys/fs/cgroup/io.pressure", "/proc/pressure/io"),
}


def parse_cpu_stat(text: str) -> Dict[str, int]:
    """`cpu.stat` → `{nr_periods, nr_throttled, throttled_usec}`.

    Pure. Normalises the two cgroup generations onto ONE unit: v2 publishes
    `throttled_usec`, v1 publishes `throttled_time` in NANOseconds, and
    reporting them in the same field without the conversion would read as a
    1000x throttling cliff the day a host moved from v1 to v2.
    """
    out: Dict[str, int] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        key, raw = parts[0], parts[1]
        try:
            val = int(raw)
        except ValueError:
            continue
        if key == "nr_periods":
            out["nr_periods"] = val
        elif key == "nr_throttled":
            out["nr_throttled"] = val
        elif key == "throttled_usec":
            out["throttled_usec"] = val
        elif key == "throttled_time":  # cgroup v1, nanoseconds
            out.setdefault("throttled_usec", val // 1000)
    return out


def parse_pressure(text: str) -> Dict[str, int]:
    """A PSI file → `{some_usec, full_usec}` (the cumulative `total=` fields).

    Only the totals: the `avgN` fields are decaying averages over windows
    nobody chose, and a DELTA of a monotonic total across a turn is the only
    thing that can be attributed to that turn. `full` is absent from
    `/proc/pressure/cpu` on many kernels — a missing key is left missing
    rather than zeroed, so "not published" and "no pressure" stay distinct.
    """
    out: Dict[str, int] = {}
    for line in (text or "").splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] not in ("some", "full"):
            continue
        for field in parts[1:]:
            if not field.startswith("total="):
                continue
            try:
                out[f"{parts[0]}_usec"] = int(field[6:])
            except ValueError:
                pass
    return out


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r") as fh:
            return fh.read(8192)
    except Exception:  # noqa: BLE001 — not Linux, not mounted, not permitted
        return None


def read_host_counters() -> Dict[str, int]:
    """Cumulative cgroup + PSI counters, or `{}`. Never raises.

    Cheap enough for twice a turn: at most five small sysfs reads, each of a
    file the kernel renders on demand in a few microseconds.
    """
    out: Dict[str, int] = {}
    try:
        for path in _CGROUP_CPU_STAT:
            if not os.path.exists(path):
                continue
            text = _read(path)
            if text:
                out.update(parse_cpu_stat(text))
                break
        for prefix, paths in _PRESSURE_FILES.items():
            for path in paths:
                if not os.path.exists(path):
                    continue
                text = _read(path)
                if not text:
                    continue
                psi = parse_pressure(text)
                if "some_usec" in psi:
                    out[f"psi_{prefix}_some_usec"] = psi["some_usec"]
                if "full_usec" in psi:
                    out[f"psi_{prefix}_full_usec"] = psi["full_usec"]
                break
    except Exception:  # noqa: BLE001
        return out
    return out


def host_delta(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    """`after - before` for every key present in BOTH. A key that appeared or
    vanished mid-turn is dropped rather than reported as a jump from zero; a
    negative delta (a counter reset, i.e. a new cgroup) is dropped too."""
    out: Dict[str, int] = {}
    for k, v in (after or {}).items():
        if k not in (before or {}):
            continue
        try:
            d = int(v) - int(before[k])
        except Exception:  # noqa: BLE001
            continue
        if d >= 0:
            out[k] = d
    return out


def emit_turn_host() -> Optional[str]:
    """One `[PERF] turn_host` line per `run()` FRAME. The line, or None.

    Per FRAME, not per turn, and the distinction is load-bearing: a turn that
    spawns a sub-agent or a mission runs a SECOND `AgentRunner.run()` in a
    context copied from this one, and that child gets its own line at
    `nest=1`, covering its own window, with its own host baseline and its own
    channel. Reading `nest=0` is how you get the line for the turn the user
    started. (Before R48's blocking finding B1 was fixed there was exactly one
    line per turn — the CHILD's, under the parent's `cmid_h` and channel,
    spanning parent-start → child-end, with the parent emitting nothing.)

    Emitted from `run()`'s `finally`, so a cancelled voice turn and a turn that
    raised are covered — those are the turns most likely to have been throttled.
    """
    t = _own_turn()
    if t is None or not t.enabled or t.host_emitted:
        return None
    t.host_emitted = True
    try:
        d = host_delta(t.host0, read_host_counters())
        line = (
            "[PERF] turn_host cpu_throttled_ms=%d nr_throttled=%d nr_periods=%d "
            "psi_cpu_some_ms=%d psi_mem_some_ms=%d psi_io_some_ms=%d "
            "src=%s nest=%d cmid_h=%s ch=%s"
        ) % (
            d.get("throttled_usec", 0) // 1000,
            d.get("nr_throttled", 0),
            d.get("nr_periods", 0),
            d.get("psi_cpu_some_usec", 0) // 1000,
            d.get("psi_mem_some_usec", 0) // 1000,
            d.get("psi_io_some_usec", 0) // 1000,
            # "none" is not a zero reading: a macOS dev box, a cgroup v1 host
            # without PSI and a genuinely unthrottled container would all print
            # zeros, and only this field tells them apart.
            "cgroup" if d else "none",
            t.nest, t.cmid_h, t.channel,
        )
        logger.info(line)
        return line
    except Exception:  # noqa: BLE001
        return None


# ── the public context manager ──────────────────────────────────────

class _NullSpan:
    """Zero-cost when the turn is not armed. Stateless, so one instance is
    reused for every no-op span — including reentrantly."""

    __slots__ = ()

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        return False


_NULL = _NullSpan()


class _ActiveSpan:
    __slots__ = ("phase", "span", "token", "_acquired")

    def __init__(self, phase: str) -> None:
        self.phase = phase
        self.span: Optional[_Span] = None
        self.token = None
        self._acquired = False

    async def __aenter__(self) -> None:
        try:
            self.span = _Span(self.phase)
            self.token = _SPAN.set(self.span)
            _lag_acquire()
            self._acquired = True
        except Exception:  # noqa: BLE001
            self.span = None
        return None

    async def __aexit__(self, *exc: Any) -> bool:
        s = self.span
        if s is None:
            # `__aenter__` failed after setting the ContextVar. The cleanup is
            # the part that MUST still run: a span left set in a long-lived
            # handler context attributes every later session to a dead phase.
            self._release()
            return False
        try:
            now = time.perf_counter()
            for key in list(s.pending):
                _close_pending(s, key, now)
            s.closed = True
            total_ms = (now - s.t0) * 1000.0
            lag = _lag_max_since(s.t0)
            seq = ",".join(s.toks) or "-"
            if s.stmts > len(s.toks):
                seq += f",+{s.stmts - len(s.toks)}"
            turn = _TURN.get()
            line = (
                "[PERF] db_span phase=%s total_ms=%d conns=%d dials=%d "
                "dial_ms=%d begin_ms=%d stmts=%d em=%d sql_ms=%d stmt1_ms=%d "
                "commit_ms=%d loop_lag_max_ms=%d lag_cov=%d slow=%s@%d seq=%s "
                "nest=%d cmid_h=%s ch=%s"
            ) % (
                s.phase, int(total_ms), s.conns, s.dials, int(s.dial_ms),
                int(s.begin_ms), s.stmts, s.em, int(s.sql_ms),
                # A SUBSET of sql_ms — never add it into an accounting sum.
                int(s.stmt1_ms), int(s.commit_ms),
                int(lag), 1 if _lag_covers(s.t0) else 0,
                # Always printed, even with no statements: "absent" and "fast"
                # must not look alike — the same rule `[PERF] ws_pre_turn`
                # already follows for its own keys.
                s.slow_tok or "-", int(s.slow_ms), seq,
                # Which `run()` frame this span belongs to. 0 is the turn the
                # user's message started; a sub-agent's spans ride the same
                # `cmid_h` at nest≥1 and would otherwise be indistinguishable
                # from the parent's (R48 blocking finding B1).
                (turn.nest if turn is not None else 0),
                turn_cmid_h(), (turn.channel if turn is not None else "-"),
            )
            logger.info(line)
        except Exception:  # noqa: BLE001 — telemetry never costs a turn
            pass
        finally:
            self._release()
        return False

    def _release(self) -> None:
        try:
            if self.token is not None:
                _SPAN.reset(self.token)
                self.token = None
        except Exception:  # noqa: BLE001 — a token from another context
            _SPAN.set(None)
            self.token = None
        # Balanced with `__aenter__`: releasing a reference that was never
        # acquired would stop the sampler out from under a sibling span.
        if self._acquired:
            self._acquired = False
            _lag_release()


def db_span(phase: str):
    """`async with db_span("save"), async_session_maker() as db:`

    The combined form is deliberate: it wraps the session's `async with`
    itself (entering first, exiting last, so the connection's checkin and the
    session close are inside the span) WITHOUT re-indenting the body, which is
    what keeps this patch a one-line change at every call site.

    A no-op — and a free one — unless `begin_turn` armed the turn.
    """
    if not turn_enabled():
        return _NULL
    return _ActiveSpan(phase)


def _reset_for_tests() -> None:
    """Test-only: this module keeps process-global state by design."""
    global _lag_task, _lag_refs, _lag_loop
    _LAG.clear()
    _reported.clear()
    if _lag_task is not None:
        try:
            _lag_task.cancel()
        except Exception:  # noqa: BLE001
            pass
    _lag_task = None
    _lag_loop = None
    _lag_refs = 0
    _SPAN.set(None)
    _TURN.set(None)
    _RUN_NEST.set(0)
