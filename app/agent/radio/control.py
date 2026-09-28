"""Awaitable media controls shared by internal callers.

The websocket handlers remain the owners of radio queue mutation and client
broadcasts.  This module gives non-websocket callers a truthful result only
after that work has completed, without duplicating those rules.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import contextvars
from dataclasses import asdict, dataclass, field, replace
import itertools
import logging
from typing import Optional
import uuid

from app.agent.radio.session import RadioSession, RadioSessionManager, get_radio_manager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MediaControlOutcome:
    ok: bool
    user_id: str
    action: str
    channel: str
    changed: bool
    reason: str
    previous_video_id: str = ""
    video_id: str = ""
    title: str = ""
    # stop/pause only: the id the device acknowledged, and what it reported.
    command_id: str = ""
    was_playing: Optional[bool] = None
    # stop/pause only, additive (review F27): how many of the sockets that
    # could be playing this channel answered, and how many stayed silent. A
    # verdict speaks for the answering devices only; `silent_devices > 0` is
    # why an all-idle answer is `partially_confirmed`, not `nothing_playing`.
    acked_devices: int = 0
    silent_devices: int = 0
    # `newer_playing` only, additive (addendum 6 R6-9 residual 4): the newer
    # item left alone is PAUSED (a pause this tenant applied to it), so the
    # relay must not say it plays. Absent from `as_dict` unless true, so no
    # other verdict changes shape.
    paused: Optional[bool] = None

    def as_dict(self) -> dict:
        data = asdict(self)
        if data.get("paused") is not True:
            data.pop("paused", None)
        return data


def media_control_budget_s() -> float:
    """The endpoint's wall-clock budget for one control (settle before the
    platform relay's 12 s HTTP timeout). One definition, because the stop/pause
    ack wait below must end INSIDE it."""

    from app.config import settings

    return max(0.1, min(9.0, float(settings.voice_live_media_control_timeout_s)))


# ── Device-confirmed stop / pause (contract v0.3 §H, `media_transport`) ──
# The station lives here, but the audio lives on the phone: a stop that only
# turned the station off left a one-off track (or the phone's own queue)
# playing, and the relay had nothing but hope to base "stopped" on. So the
# phone is asked over the chat socket and its ack is the only thing that makes
# `ok` true. Acks arrive on a DIFFERENT task (the socket's receive loop, or its
# mid-turn stop-watcher), hence a registry of futures keyed by command id.
#
# Bounded twice: every entry is removed by its own waiter when the wait ends
# (ack, deadline or cancellation), and the map is capped so a burst cannot grow
# it — the oldest waiter is answered with the acks it has (usually none, i.e.
# `unacknowledged`) instead of being left to hang.
_TRANSPORT_FRAMES = {
    "stop": ("media_stop", "media_stop_ack"),
    "pause": ("media_pause", "media_pause_ack"),
}
MEDIA_ACK_TYPES = frozenset(ack for _frame, ack in _TRANSPORT_FRAMES.values())
_MAX_PENDING_ACKS = 32
_MAX_COMMAND_ID_LEN = 64
# Room left inside the endpoint budget for the answer to get out after the ack
# wait gives up, so an unanswered stop reports `unacknowledged` rather than the
# endpoint's own `timeout`.
_ACK_MARGIN_S = 1.0
# How long the other sockets get once one device has answered without stopping
# anything (idle, already stopped, or kept playing). Most chat sockets cannot
# answer at all: the web ChatPage drops channel-'app' frames, the desktop
# bridge and the browser extension have no media_stop handler, a half-open
# phone queue answers nothing. Waiting for them until the budget ran out held
# the call silent for 7 s (review F27). A socket KNOWN to answer (it answered
# an earlier command on this connection and is live, `ws_chat`'s socket
# evidence) is still waited for until the deadline: its answer is coming, and
# a player that answers late must still make it `stopped`.
_PEER_ACK_GRACE_S = 1.5
# The verdict when every device that answered was idle but a socket that could
# be playing this channel stayed silent. The station is OFF (stop) either way;
# what cannot be said is "nothing was playing" — that is only the answering
# devices' report, and an idle socket answering first must not speak for a
# silent one that may be the player (review F27 residual, verifier case D).
PARTIAL_REASON = "partially_confirmed"


@dataclass
class _PendingAck:
    user_id: str
    ack_type: str
    future: "asyncio.Future"
    # How many acks the verdict must cover: every socket the command reached
    # that could be playing its channel (known answerers + unknown sockets).
    expected: int = 1
    acks: list = field(default_factory=list)
    grace: Optional["asyncio.TimerHandle"] = None
    # Reached sockets (queues, by identity) that are known to answer and have
    # not yet: while one of these is silent the idle answers get no grace.
    awaiting: set = field(default_factory=set)
    # Reached sockets proven unable to play the command's channel (a web tab
    # that declared channel 'web', the extension). Not in `expected`.
    exempt: set = field(default_factory=set)
    # Sockets already heard from: the same socket answering twice counts once.
    heard: set = field(default_factory=set)
    # Acks that arrived with no socket attached (direct/internal callers).
    anonymous: int = 0
    timed_out: bool = False


_pending_acks: "OrderedDict[str, _PendingAck]" = OrderedDict()


def _default_ack_timeout_s() -> float:
    budget = media_control_budget_s()
    return max(0.0, budget - min(_ACK_MARGIN_S, budget * 0.25))


# ── A play asked for before a stop must not start after it ────────────────
# A play's search takes 0.5–2 s and up to ~25 s; a spoken stop is an OFF, a
# frame and an ack. So "play some jazz" then "stop the music" can end with the
# stop confirmed FIRST and the older play's `media_play` landing after it. The
# phone reads any non-auto media_play as an explicit user play: it lifts the
# user-stop fence, starts the stale track and re-seeds the station the stop
# had just turned off (review F29). This tenant sends that frame, so the order
# is decided here: a play takes a mark when it is asked for and re-reads it,
# synchronously, right before it broadcasts (`media_play_superseded`).
#
# Two things count as a newer halt:
#   * a voice stop/pause accepted by `_execute_transport`. A pause leaves the
#     station alone, so it needs its own sequence;
#   * any radio_toggle OFF: the voice stop's own, the player's X on the phone,
#     the station pill, the web player's close. That is
#     `ws_chat._radio_off_generations`, the primitive that already drops a
#     queued ON behind a newer OFF.
# Both are read per USER, not per channel: media_play goes to every socket the
# user has, so a stop on any of them makes an older play stale on all of them.
#
# ── Order-keyed halts (contract v0.3 §7, addendum 6 R6-4 / R6-4b) ─────────
# Arrival order is not the caller's order. The relay holds a spoken stop for a
# grace before it fires, and an agent run that is still THINKING took its mark
# when it arrived — so "play Halo" → "stop the music" → "play some jazz" could
# reach this tenant as play(Halo), play(jazz), stop, and the arrival rule then
# dropped the jazz the caller asked for AFTER the stop (media-1, media-4), or
# the relay withheld the stop and the older Halo played on (media-3).
#
# So a relay that knows the caller's order says it: a play / run carries
# `media_order` (the caller-turn ordinal that asked for it) and `media_scope`
# (the provider session), a stop/pause carries `before_order` and the same
# scope, and each may carry `media_scope_started_ms` — the scope STAMP, a
# hybrid logical clock the relay derives from the app's floor (R6-4b), so a
# newer provider session of one call always carries a larger stamp whichever
# relay replica served it.
#
# SKEW-SAFE, PAIRWISE (R6-4b; replaces the single total key and the tenant
# clock of the first R6 cut): two ordered requests compare
#   * in the SAME scope by their orders (no stamp needed);
#   * across scopes, when BOTH carry a stamp, by their stamps — equal stamps
#     (only causally unrelated scopes, e.g. two devices) break on the order in
#     which this tenant first saw the two scopes (a sequence, not a clock);
#   * across scopes when EITHER carries no stamp: causally incomparable, so
#     the arrival rule decides that pair — exactly the pre-R6 semantics.
# No clock of this tenant is ever read for ordering: a relay stamp is never
# compared with anything but another relay stamp.
#
# Halts are kept per scope (its largest-order halt and its newest-arriving
# one), for the newest `_MAX_SCOPES` scopes of the user. A play is superseded
# iff SOME recorded halt beats it under the pairwise rule. The device-stop
# gate (`newer_playing`) and the play-vs-broadcast fence (R6-9 T1) use the same
# rule.
#
# Whenever either side carries no order — the phone's X, a typed stop, a
# legacy relay, the TF132 paths — the arrival rule above applies unchanged
# (addendum 2 item 9): an ordered halt still counts, by arrival, against an
# unordered play, and an unordered halt against an ordered one. The radio OFF
# an ORDERED stop makes is its own act (the halt itself is recorded and
# compared pairwise), so it is excluded from what an ordered play compares by
# arrival (`_UserMediaOrder.ordered_offs`); every other OFF still counts.
_halt_seq = itertools.count(1)
_scope_seq = itertools.count(1)
# user_id -> `_UserMediaOrder`. One entry per user, and a tenant has one user;
# the cap is a guard, not a working set. An evicted entry only loses halts
# older than every play still in flight.
_latest_halts: "OrderedDict[str, _UserMediaOrder]" = OrderedDict()
_MAX_HALT_USERS = 256
# Scopes kept per user (one scope = one provider session), newest by use. A
# reconnect is a new scope; the newest few dozen are far more than one call
# can reach.
_MAX_SCOPES = 32
_MAX_SCOPE_LEN = 128
_MAX_MEDIA_ORDER = 2**31 - 1
# A usable stamp is an int (never a bool) with 0 < stamp < 2**53 — the same
# bound the relay applies to the app's floor.
_MAX_STARTED_MS = 2**53
# The prefix `_tool_play_media` answers with when its play was superseded. A
# decision, not a failure: never "ERROR", so it is not a failed step and the
# model is not invited to retry it. `internal_play_media` maps it to
# {"ok": false, "reason": "superseded"}.
PLAY_SUPERSEDED_PREFIX = "SUPERSEDED:"
# The stop/pause verdict when a NEWER ordered item is what the tenant last
# broadcast (contract v0.3 §7, additive, ok=false): the halt was requested
# before that item, so it does not touch it — no station OFF, no media_stop /
# media_pause frame. The relay words it as "the newer request is what plays",
# never "stopped".
NEWER_PLAYING_REASON = "newer_playing"
# What `media_play_superseded` answers when no halt but the item already on
# the device supersedes the play (R6-9 T1): the caller asked for that item
# AFTER this play, so this older play must not replace it.
NEWER_PLAY_SUPERSEDES = "newer_play"


@dataclass(frozen=True)
class _OrderedEvent:
    """One ordered request as the pairwise rule sees it."""

    scope: str
    order: int
    # The relay's scope stamp this request carried, or None (then it compares
    # with other scopes by arrival).
    stamp: Optional[int]
    # When this tenant first saw the scope (a sequence): breaks equal stamps.
    first_seen: int
    # Arrival: this tenant's halt sequence (the request's own for a halt; the
    # newest halt it had seen for a play/item).
    seq: int


def _caller_order(a: _OrderedEvent, b: _OrderedEvent) -> Optional[int]:
    """1 when the caller asked for `a` after `b`, -1 before, 0 in the same
    turn; None when the pair cannot be ordered by the caller's order (different
    scopes, either unstamped) and the arrival rule decides it."""

    if a.scope == b.scope:
        return (a.order > b.order) - (a.order < b.order)
    if a.stamp is None or b.stamp is None:
        return None
    ka, kb = (a.stamp, a.first_seen), (b.stamp, b.first_seen)
    return (ka > kb) - (ka < kb)


def _halt_beats(halt: _OrderedEvent, play: _OrderedEvent) -> bool:
    """A halt supersedes a play it was asked for after; an incomparable pair
    falls back to arrival (the halt arrived after the play's mark)."""

    order = _caller_order(halt, play)
    if order is None:
        return halt.seq > play.seq
    return order > 0


@dataclass(frozen=True)
class MediaHaltMark:
    """What a play request saw when it began. Only this module makes one, and
    a tool input coming from a model is JSON, so a model cannot forge it.

    ``media_order`` / ``media_scope`` (R6) are the caller's order for the
    request, when the relay sent one; both or neither (`ordered`), and
    ``media_scope_started_ms`` its scope stamp when it carried a usable one.
    ``halt_seq`` is its arrival (the newest halt already seen). For an
    ordered mark ``unordered_off`` is what the arrival rule compares for radio
    OFFs: only the OFFs that did not come from an ordered stop."""

    user_id: str
    halt_seq: int
    off_generation: int
    media_order: Optional[int] = None
    media_scope: str = ""
    media_scope_started_ms: Optional[int] = None
    scope_first_seen: int = 0
    unordered_halt_seq: int = 0
    unordered_off: int = 0

    @property
    def ordered(self) -> bool:
        return self.media_order is not None and bool(self.media_scope) and self.scope_first_seen > 0

    def _event(self) -> Optional[_OrderedEvent]:
        if not self.ordered:
            return None
        return _OrderedEvent(
            scope=self.media_scope,
            order=int(self.media_order),
            stamp=self.media_scope_started_ms,
            first_seen=self.scope_first_seen,
            seq=self.halt_seq,
        )


@dataclass(frozen=True)
class _Halt:
    event: _OrderedEvent
    action: str


@dataclass
class _ScopeHalts:
    """One scope of the user: when this tenant first saw it, and its halts —
    the largest-order one (same-scope order) and the newest-arriving one
    (arrival pairs)."""

    first_seen: int
    top: Optional[_Halt] = None
    last: Optional[_Halt] = None

    def halts(self) -> list:
        found = [h for h in (self.last, self.top) if h is not None]
        return found if len(found) < 2 or found[0] is not found[1] else found[:1]


@dataclass(frozen=True)
class _Broadcast:
    """The item this tenant last broadcast as a media_play for the user, and
    the caller's order for it (None when the play carried no order). A
    station re-announcement of the same item keeps that order (R6-10 TA1)."""

    event: Optional[_OrderedEvent]
    video_id: str
    title: str
    # `_unordered_off_total` when it was broadcast: an OFF that did not come
    # from an ordered stop (the phone's X, the pill, a web close) stopped it.
    unordered_off: int
    # A pause this tenant applied to it (R6-9 residual 4).
    paused: bool = False


@dataclass
class _UserMediaOrder:
    """Per-user halt bookkeeping (the values of `_latest_halts`)."""

    # The newest halt of any kind: the arrival rule for an UNORDERED play.
    halt_seq: int = 0
    halt_action: str = ""
    # The newest halt that carried no order: the arrival rule for an ORDERED
    # play (an ordered halt is compared pairwise instead).
    unordered_seq: int = 0
    unordered_action: str = ""
    # Radio OFF generations bumped by ORDERED stops (their own OFF).
    ordered_offs: int = 0
    # scope -> its halts and first-seen sequence, newest by use last.
    scopes: "OrderedDict[str, _ScopeHalts]" = field(default_factory=OrderedDict)
    # What was last broadcast, or None once a stop was applied to it.
    playing: Optional[_Broadcast] = None


def media_order_scope(order: object, scope: object) -> tuple:
    """(order, scope) when both are usable, else (None, "").

    An order is a non-negative int (a digit string is accepted; a bool is
    not); a scope is a non-empty string, trimmed and capped. An order without
    a scope compares with nothing, so it is treated as no order at all: the
    request then takes the unchanged arrival rule."""

    order = _plain_int(order, _MAX_MEDIA_ORDER)
    if order is None or not isinstance(scope, str) or not scope.strip():
        return None, ""
    return order, scope.strip()[:_MAX_SCOPE_LEN]


def media_scope_stamp(value: object) -> Optional[int]:
    """A usable scope stamp (int, not bool, 0 < v < 2**53; a digit string is
    accepted like the order), else None — the request is then unstamped."""

    stamp = _plain_int(value, _MAX_STARTED_MS - 1)
    return stamp if stamp else None


def _plain_int(value: object, ceiling: int) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if isinstance(value, int) and 0 <= value <= ceiling:
        return int(value)
    return None


def _user_state(user_id: str, *, create: bool = False) -> Optional[_UserMediaOrder]:
    state = _latest_halts.get(user_id)
    if not isinstance(state, _UserMediaOrder):
        if not create:
            return None
        state = _UserMediaOrder()
        _latest_halts[user_id] = state
    if create:
        _latest_halts.move_to_end(user_id)
        while len(_latest_halts) > _MAX_HALT_USERS:
            _latest_halts.popitem(last=False)
    return state


def _scope_record(state: _UserMediaOrder, scope: str) -> _ScopeHalts:
    """The scope's record, created (first-seen now) when new; kept newest by
    use and bounded to `_MAX_SCOPES` (an evicted scope takes its halts)."""

    record = state.scopes.get(scope)
    if record is None:
        record = _ScopeHalts(first_seen=next(_scope_seq))
        state.scopes[scope] = record
    state.scopes.move_to_end(scope)
    while len(state.scopes) > _MAX_SCOPES:
        state.scopes.popitem(last=False)
    return record


def _ordered_event(
    user_id: str, order: object, scope: object, started_ms: object, *, seq: int,
) -> Optional[_OrderedEvent]:
    """The pairwise-rule view of an ordered request, or None when it carries
    no usable order + scope (it is then unordered: arrival rule)."""

    order, scope = media_order_scope(order, scope)
    if order is None or not user_id:
        return None
    state = _user_state(user_id, create=True)
    record = _scope_record(state, scope)
    return _OrderedEvent(
        scope=scope,
        order=order,
        stamp=media_scope_stamp(started_ms),
        first_seen=record.first_seen,
        seq=seq,
    )


def _unordered_off_total(user_id: str, state: Optional[_UserMediaOrder] = None) -> int:
    """The user's radio OFF count minus the OFFs ordered stops made."""

    if state is None:
        state = _user_state(user_id)
    return _off_generation_total(user_id) - (state.ordered_offs if state else 0)


def _off_generation_total(user_id: str) -> int:
    # Read at call time, by attribute: ws_chat owns the map and tests swap it.
    try:
        from app.api import ws_chat

        generations = getattr(ws_chat, "_radio_off_generations", None) or {}
        return sum(
            int(count or 0)
            for key, count in list(generations.items())
            if isinstance(key, tuple) and key[:1] == (user_id,)
        )
    except Exception:  # noqa: BLE001 - a guard must never break a play
        return 0


def media_halt_mark(
    user_id: str,
    *,
    media_order: object = None,
    media_scope: object = None,
    media_scope_started_ms: object = None,
) -> MediaHaltMark:
    """Take the mark a play request is later checked against.

    ``media_order`` / ``media_scope`` (+ ``media_scope_started_ms``) are the
    relay's caller order for the request (contract v0.3 §7); without an order
    and a scope the mark is unordered and behaves exactly as before."""

    uid = str(user_id or "")
    state = _user_state(uid)
    seq = state.halt_seq if state else 0
    event = _ordered_event(uid, media_order, media_scope, media_scope_started_ms, seq=seq)
    state = _user_state(uid)
    return MediaHaltMark(
        user_id=uid,
        halt_seq=seq,
        off_generation=_off_generation_total(uid),
        media_order=event.order if event else None,
        media_scope=event.scope if event else "",
        media_scope_started_ms=event.stamp if event else None,
        scope_first_seen=event.first_seen if event else 0,
        unordered_halt_seq=state.unordered_seq if state else 0,
        unordered_off=_unordered_off_total(uid, state),
    )


def media_play_superseded(mark: Optional[MediaHaltMark]) -> Optional[str]:
    """'stop' or 'pause' when a halt supersedes the play `mark` was taken for,
    `NEWER_PLAY_SUPERSEDES` when the item already broadcast does, else None.

    Unordered mark: a halt of any kind that landed after the mark (arrival).
    Ordered mark: a halt with no order that landed after it (arrival), or an
    ordered halt that beats it under the pairwise rule (same scope: a larger
    order; both stamped: a larger stamp, first-seen on a tie; otherwise:
    arrival), in any scope, whenever it landed; or the recorded broadcast item
    when the caller asked for it after this play (R6-9 T1) and nothing without
    an order stopped it since — the caller's newest request stays on.

    Synchronous on purpose: the caller runs it with no await between it and
    its media_play broadcast. A stop that lands after the broadcast still sends
    its media_stop after the media_play, so the newer intent wins either way."""

    if not isinstance(mark, MediaHaltMark) or not mark.user_id:
        return None
    state = _user_state(mark.user_id)
    play = mark._event()
    if play is None:
        if state and state.halt_seq > mark.halt_seq:
            return state.halt_action
        if _off_generation_total(mark.user_id) > mark.off_generation:
            return "stop"
        return None
    if state and state.unordered_seq > mark.halt_seq:
        return state.unordered_action or "stop"
    if _unordered_off_total(mark.user_id, state) > mark.unordered_off:
        return "stop"
    if state is None:
        return None
    beaten = [
        halt
        for record in list(state.scopes.values())
        for halt in record.halts()
        if _halt_beats(halt.event, play)
    ]
    if beaten:
        return max(beaten, key=lambda h: h.event.seq).action or "stop"
    item = _live_item(mark.user_id, state)
    if item is not None and item.event is not None:
        order = _caller_order(item.event, play)
        # Only a caller-order win: an equal key is the same turn, and an
        # incomparable pair keeps the pre-R6 arrival semantics (a play never
        # fenced another play by arrival).
        if order is not None and order > 0:
            return NEWER_PLAY_SUPERSEDES
    return None


def _note_media_halt(user_id: str, action: str, *, halt: Optional[_OrderedEvent] = None) -> None:
    state = _user_state(user_id, create=True)
    seq = next(_halt_seq)
    state.halt_seq, state.halt_action = seq, action
    if halt is None:
        state.unordered_seq, state.unordered_action = seq, action
        return
    record = _scope_record(state, halt.scope)
    recorded = _Halt(
        event=_OrderedEvent(
            scope=halt.scope, order=halt.order, stamp=halt.stamp,
            first_seen=record.first_seen, seq=seq,
        ),
        action=action,
    )
    record.last = recorded
    if record.top is None or halt.order >= record.top.event.order:
        record.top = recorded


def _live_item(user_id: str, state: Optional[_UserMediaOrder]) -> Optional[_Broadcast]:
    """The recorded broadcast item, unless an OFF without an order (the
    phone's X, the pill, a web close) stopped it since it was broadcast."""

    item = state.playing if state else None
    if item is None:
        return None
    if _unordered_off_total(user_id, state) > item.unordered_off:
        return None
    return item


def note_media_broadcast(
    user_id: str,
    mark: Optional[MediaHaltMark] = None,
    *,
    video_id: str = "",
    title: str = "",
    reannounces: str = "",
) -> None:
    """Record what this tenant just broadcast as a media_play, with the
    caller's order for it when its mark carried one (R6), unordered
    otherwise. The device stop and the play fence read it.

    ``reannounces`` (addendum 6 R6-10 TA1): the video id of the item this
    frame RE-ANNOUNCES — the station's toggle seed answering the app's reseed
    of the song it is playing, a re-anchor to the station's current track, a
    variant swap (same song, other surface) of the current track. When that is
    the item recorded as live, the frame is the SAME item: it keeps the
    caller's order, scope and stamp (its video id and title become the
    frame's). Anything else — no `reannounces`, or one that names a different
    item (a radio tap on an older card), or a live item already stopped — is a
    new item, unordered unless its own mark carries an order. So the caller's
    newest request keeps its place however often the station re-announces it,
    and a genuinely different station track (next, auto-advance, a history
    step, a saved playlist) is still a new unordered item an ordered stop
    reaches.

    Called only by `send_media_play`, the one chokepoint every media_play
    leaves through (R6-9 T4), right after the frame reached a socket. Never
    raises into a play."""

    uid = str(user_id or "")
    if not uid:
        return
    try:
        state = _user_state(uid, create=True)
        event = None
        if isinstance(mark, MediaHaltMark) and mark.user_id == uid:
            event = mark._event()
        if event is None and reannounces:
            live = _live_item(uid, state)
            if live is not None and live.video_id and live.video_id == str(reannounces)[:64]:
                event = live.event
        if event is not None and event.scope in state.scopes:
            # The scope's first-seen as recorded now (the mark's is the same
            # unless the scope was evicted and seen again).
            event = _OrderedEvent(
                scope=event.scope, order=event.order, stamp=event.stamp,
                first_seen=state.scopes[event.scope].first_seen, seq=event.seq,
            )
        state.playing = _Broadcast(
            event=event,
            video_id=str(video_id or "")[:64],
            title=str(title or "")[:200],
            unordered_off=_unordered_off_total(uid, state),
        )
    except Exception:  # noqa: BLE001 - bookkeeping must never break a play
        logger.debug("[media-control] broadcast bookkeeping failed", exc_info=True)


def holds_paused_reannouncement(
    user_id: str, reannounces: str, mark: Optional[MediaHaltMark] = None,
) -> bool:
    """R6-11 TB4: True when a station frame that RE-ANNOUNCES an item (the
    toggle seed after the app's reseed build, a re-anchor, a variant swap —
    `reannounces`, with no ordered mark of its own) names the item recorded as
    live and that item is PAUSED.

    A pause is recorded only on the very item the device paused, after that
    item was broadcast — so the halt that paused it is newer than it (by the
    caller's order, or by arrival). A media_play of the same item resumes it
    on the phone ('same track — resume if paused'), which would undo the
    caller's newer request with a frame caused by the older one. So it is
    held: the station ships its state and window only, and the record stays
    paused. A new play (its own mark, or no `reannounces`) is a request of its
    own and is never held."""

    uid = str(user_id or "")
    item_id = str(reannounces or "")[:64]
    if not uid or not item_id:
        return False
    if isinstance(mark, MediaHaltMark) and mark._event() is not None:
        return False
    live = _live_item(uid, _user_state(uid))
    return bool(live is not None and live.paused and live.video_id == item_id)


async def send_media_play(
    user_id: str,
    frame: dict,
    *,
    mark: Optional[MediaHaltMark] = None,
    queue: "Optional[asyncio.Queue]" = None,
    reannounces: str = "",
) -> int:
    """Send one media_play frame and record it as what plays (R6-9 T4).

    Every media_play this tenant sends — the play tool (voice and agent), the
    typed-chat fast path (`queue`: the sender's own socket) and every station
    broadcast (`radio.player.broadcast_radio_track`: next, auto-advance, the
    toggle seed, a surface swap, a saved playlist) — leaves through here, so
    the device gate and a `newer_playing` verdict read what actually plays.
    `mark` gives the item the caller's order; without an ordered mark the item
    is unordered — unless the frame `reannounces` the item already recorded
    as live (the station's re-announcement of the same item, TA1: it keeps
    that item's causal identity; see `note_media_broadcast`). A
    re-announcement of an item the caller PAUSED is never sent (R6-11 TB4,
    `holds_paused_reannouncement`): 0, nothing recorded, the item stays
    paused. Returns how many sockets got the frame; nothing is recorded when
    none did. No await before the send, so a caller's synchronous supersede
    check stays adjacent to it."""

    if reannounces and holds_paused_reannouncement(user_id, reannounces, mark):
        logger.info(
            "[media-control] re-announcement held: item paused by a newer halt user=%s",
            str(user_id or "")[:8],
        )
        return 0
    if queue is not None:
        queue.put_nowait(frame)
        sent = 1
    else:
        from app.api.ws_chat import broadcast_to_user

        sent = await broadcast_to_user(user_id, frame)
    if sent:
        note_media_broadcast(
            user_id, mark,
            video_id=str(frame.get("video_id") or ""),
            title=str(frame.get("title") or ""),
            reannounces=str(reannounces or ""),
        )
    return sent


def recorded_media_item(user_id: str) -> Optional[dict]:
    """{video_id, title, paused} of the item recorded as playing, or None."""

    state = _user_state(str(user_id or ""))
    item = _live_item(str(user_id or ""), state)
    if item is None:
        return None
    return {"video_id": item.video_id, "title": item.title, "paused": item.paused}


def _newer_playing(user_id: str, halt: Optional[_OrderedEvent]) -> Optional[_Broadcast]:
    """The broadcast item an ordered halt must leave alone, or None.

    Only an ordered item the caller asked for after the halt, or in its own
    turn, under the pairwise rule (same scope: order; both stamped: stamp,
    then first-seen) — and that no OFF without an order has stopped since. An
    item with no order, an older one, one already stopped, or one the halt
    cannot be ordered against (different scopes, either unstamped: the halt
    arrived after its broadcast) is stopped."""

    if halt is None:
        return None
    state = _user_state(user_id)
    item = _live_item(user_id, state)
    if item is None or item.event is None:
        return None
    order = _caller_order(item.event, halt)
    return item if order is not None and order >= 0 else None


# ── The run-level mark: a stop while the agent is still THINKING ──────────
# The agent's play_media tool used to take its mark when the tool started. A
# stop that landed earlier — while the model was still deciding, before it
# called play_media — was older than that mark and so never superseded the
# play (fx2-tenant not_done). The request's own arrival is the moment that
# counts, so the entry points that start an agent run for a user request
# (the voice `think` endpoints, a typed chat turn) take the mark on arrival
# and bind it here for the run; the tool reads it before taking its own.
# A ContextVar: per asyncio task and copied into the tasks a run spawns, so a
# mark bound for one run can never be read by a concurrent one.
_RUN_HALT_MARK: "contextvars.ContextVar[Optional[MediaHaltMark]]" = contextvars.ContextVar(
    "toup_media_run_halt_mark", default=None,
)


def bind_run_halt_mark(mark: Optional[MediaHaltMark]) -> "contextvars.Token":
    """Bind `mark` for the agent run about to start in this context. Returns
    the token for `reset_run_halt_mark`. A non-mark binds nothing."""

    return _RUN_HALT_MARK.set(mark if isinstance(mark, MediaHaltMark) else None)


def reset_run_halt_mark(token: "contextvars.Token") -> None:
    try:
        _RUN_HALT_MARK.reset(token)
    except (ValueError, RuntimeError):  # a different context: clear instead
        _RUN_HALT_MARK.set(None)


def run_halt_mark(user_id: str) -> Optional[MediaHaltMark]:
    """The mark bound for the current run, when it is this user's."""

    mark = _RUN_HALT_MARK.get()
    if isinstance(mark, MediaHaltMark) and mark.user_id and mark.user_id == str(user_id or ""):
        return mark
    return None


def deliver_media_ack(user_id: str, msg: dict, source: object = None) -> bool:
    """Route a phone's `media_stop_ack` / `media_pause_ack` to its command.

    True when the ack belongs to a live command of THIS user (a late duplicate
    included). An unknown id, another user's command, or an ack of the other
    kind settles nothing.

    `source` is the answering socket's broadcast queue when the chat socket
    knows it (`ws_chat._handle_media_ack`): the same socket answering twice
    then counts once, and a known answerer that has answered stops holding the
    idle answers back. Without one the ack is counted as it comes."""

    command_id = msg.get("command_id")
    if not isinstance(command_id, str) or not (0 < len(command_id) <= _MAX_COMMAND_ID_LEN):
        return False
    pending = _pending_acks.get(command_id)
    if pending is None or pending.user_id != user_id or pending.ack_type != msg.get("type"):
        return False
    if pending.future.done():
        return True
    if source is not None:
        if source in pending.heard:
            return True  # a repeat from the same socket is not a second device
        pending.heard.add(source)
        pending.awaiting.discard(source)
        if source in pending.exempt:
            # It answered after all, so it is a device: count it.
            pending.exempt.discard(source)
            pending.expected += 1
    else:
        pending.anonymous += 1
    done_key = "paused" if pending.ack_type == "media_pause_ack" and "paused" in msg else "stopped"
    done = msg.get(done_key) is True
    was_playing = msg.get("was_playing")
    if not isinstance(was_playing, bool):
        was_playing = None
    pending.acks.append((done, was_playing))
    # One device that stopped real playback settles it at once. Otherwise wait
    # for every socket that could be playing: with a phone and a second
    # socket, an idle one answering "nothing was playing" first must not hide
    # the player. A socket known to answer is waited for until the deadline;
    # the others (which may never answer) get `_PEER_ACK_GRACE_S` after this
    # answer, and if they stay silent the verdict says so (PARTIAL_REASON).
    if (done and was_playing is not False) or len(pending.acks) >= pending.expected:
        _settle_with_acks(pending)
    elif pending.grace is None and len(pending.awaiting) <= pending.anonymous:
        pending.grace = pending.future.get_loop().call_later(
            _PEER_ACK_GRACE_S, _settle_with_acks, pending,
        )
    return True


def _settle_with_acks(pending: _PendingAck) -> None:
    """Answer a waiter with the acks it has (possibly none)."""
    if not pending.future.done():
        pending.future.set_result(list(pending.acks))


def _register_ack(command_id: str, user_id: str, ack_type: str) -> _PendingAck:
    while len(_pending_acks) >= _MAX_PENDING_ACKS:
        _old_id, old = _pending_acks.popitem(last=False)
        if not old.future.done():
            old.future.set_result(list(old.acks))
    pending = _PendingAck(
        user_id=user_id,
        ack_type=ack_type,
        future=asyncio.get_running_loop().create_future(),
    )
    _pending_acks[command_id] = pending
    return pending


def _aim(pending: _PendingAck, audience: list, sent: int) -> None:
    """Scope a command to the sockets it reached (review F27).

    `audience` is `ws_chat.media_command_audience` taken with no await before
    the broadcast, as (queue, kind) pairs. When the broadcast reached exactly
    those sockets, a socket proven unable to play the channel is left out of
    what the verdict must cover and a live known answerer is awaited by name.
    When the counts disagree (a full queue dropped the frame) nobody can say
    which sockets got it, so every recipient counts and none is awaited."""

    if len(audience) == sent:
        for queue, kind in audience:
            if kind == "exempt":
                pending.exempt.add(queue)
            elif kind == "answerer":
                pending.awaiting.add(queue)
    pending.expected = max(0, int(sent) - len(pending.exempt))


def _ack_verdict(action: str, acks: list, silent: int = 0) -> tuple:
    """(ok, reason, was_playing) from what the devices answered.

    `silent` counts the sockets that could be playing this channel and did not
    answer. It changes only the all-idle verdict: an idle answer speaks for
    its own device, so with a silent one beside it the truth is partial."""

    if not acks:
        return False, "unacknowledged", None
    # `was_playing` is reported, never inferred: None when no device said.
    stopped = [wp for done, wp in acks if done and wp is not False]
    if stopped:
        return True, ("stopped" if action == "stop" else "paused"), (True in stopped) or None
    kept = [wp for done, wp in acks if not done and wp is not False]
    if kept:
        # A device that was (or may have been) playing and did not stop.
        return False, "error", (True in kept) or None
    if silent > 0:
        # Every answer was "nothing playing here", and a socket that may be
        # the player said nothing. Not ok, and never "nothing was playing".
        return False, PARTIAL_REASON, None
    return True, "nothing_playing", False


async def _execute_transport(
    user_id: str,
    action: str,
    channel: str,
    ack_timeout_s: float,
    *,
    before_order: object = None,
    media_scope: object = None,
    media_scope_started_ms: object = None,
) -> MediaControlOutcome:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + max(0.0, ack_timeout_s)
    frame_type, ack_type = _TRANSPORT_FRAMES[action]
    command_id = str(uuid.uuid4())
    before = get_radio_manager().get(user_id, channel)
    previous_video_id = _video_id(before)

    counts = {"acked": 0, "silent": 0}

    def result(ok: bool, reason: str, was_playing: Optional[bool] = None) -> MediaControlOutcome:
        return MediaControlOutcome(
            ok=ok,
            user_id=user_id,
            action=action,
            channel=channel,
            changed=bool(ok and reason in {"stopped", "paused"}),
            reason=reason,
            previous_video_id=previous_video_id,
            command_id=command_id,
            was_playing=was_playing,
            acked_devices=counts["acked"],
            silent_devices=counts["silent"],
        )

    if not RadioSessionManager.is_channel_allowed(channel):
        # The relay always sends 'app'. Anything else is a caller bug, and the
        # answer stays inside the closed stop/pause reason set it words from.
        logger.warning("[media-control] %s rejected channel=%r", action, channel[:32])
        return result(False, "error")

    # Before anything awaits: from here on, a play asked for earlier is stale,
    # whatever the phone later answers (`media_play_superseded`). An ORDERED
    # halt (R6) makes stale exactly the ordered plays it beats under the
    # pairwise rule, in any scope; by arrival it still counts against plays
    # that carry no order.
    halt = _ordered_event(user_id, before_order, media_scope, media_scope_started_ms, seq=0)
    newer = _newer_playing(user_id, halt)
    _note_media_halt(user_id, action, halt=halt)
    if newer is not None:
        # The item the tenant last broadcast was requested AFTER this halt
        # (or in its own turn): the caller's newest request wins. No station
        # OFF, no frame — the halt only drops the older plays still in
        # flight, above. Named, so the relay never says "stopped"; `paused`
        # when that item is paused, so it never says "plays" either.
        logger.info(
            "[media-control] %s user=%s order=%s left newer item order=%s paused=%s",
            action, user_id[:8], halt.order if halt else None,
            newer.event.order if newer.event else None, newer.paused,
        )
        return MediaControlOutcome(
            ok=False,
            user_id=user_id,
            action=action,
            channel=channel,
            changed=False,
            reason=NEWER_PLAYING_REASON,
            previous_video_id=previous_video_id,
            video_id=newer.video_id,
            title=newer.title,
            paused=True if newer.paused else None,
        )
    state = _user_state(user_id)
    # What was broadcast when this halt was applied: a pause the device
    # confirms marks exactly that item paused (not one broadcast meanwhile).
    halted_item = state.playing if state is not None else None
    if action == "stop" and state is not None:
        # Whatever was broadcast is being stopped: it is no longer an item a
        # later halt could leave alone. A pause keeps it (paused, not gone).
        state.playing = None

    from app.api import ws_chat

    try:
        if action == "stop":
            # Exactly the user's radio_toggle OFF: it bumps the OFF generation
            # (abandoning queued ONs) and the station epoch (invalidating an
            # in-flight refill/advance), and works when no station exists —
            # the phone may be playing a one-off track with no station behind
            # it. Done FIRST so nothing the station sends can restart music
            # between the phone stopping and the station noticing.
            #
            # An ORDERED stop's OFF is this halt's own act: credited before
            # the toggle bumps the generation (synchronously, before its first
            # await), so an ordered play compares only the OFFs that carried
            # no order. If the toggle made no OFF at all, the credit is undone.
            off_before = _off_generation_total(user_id)
            if halt is not None:
                _user_state(user_id, create=True).ordered_offs += 1
            try:
                await ws_chat._handle_radio_toggle(
                    user_id, {"channel": channel, "enabled": False, "reason": "voice"},
                )
            finally:
                if halt is not None and _off_generation_total(user_id) <= off_before:
                    state = _user_state(user_id)
                    if state is not None and state.ordered_offs > 0:
                        state.ordered_offs -= 1
        pending = _register_ack(command_id, user_id, ack_type)
        try:
            # Which sockets this frame is about to reach and what each has
            # shown it is. No await between this and the broadcast, so the
            # two describe the same set of sockets.
            audience = _media_audience(ws_chat, user_id)
            sent = await ws_chat.broadcast_to_user(user_id, {
                "type": frame_type,
                "channel": channel,
                "command_id": command_id,
                "reason": "voice",
            })
            if sent:
                _aim(pending, audience, int(sent))
            if not sent or pending.expected <= 0:
                # Nothing that can play this channel got it: no socket at all,
                # or only ones that cannot play it (a web tab, the extension).
                verdict = (False, "delivery_failed", None)
            else:
                try:
                    await asyncio.wait_for(
                        asyncio.shield(pending.future),
                        timeout=max(0.0, deadline - loop.time()),
                    )
                except asyncio.TimeoutError:
                    pending.timed_out = True
                # Freeze: an ack from here on is a straggler and counts nowhere.
                _settle_with_acks(pending)
                acks = list(pending.acks)
                silent = max(0, pending.expected - len(acks))
                counts["acked"], counts["silent"] = len(acks), silent
                verdict = _ack_verdict(action, acks, silent)
                if pending.timed_out and pending.awaiting and not pending.anonymous:
                    # A socket that had answered before stayed silent to the
                    # deadline: until it answers again it is not waited for.
                    _note_missed(ws_chat, pending.awaiting)
        finally:
            if pending.grace is not None:
                pending.grace.cancel()
            if _pending_acks.get(command_id) is pending:
                del _pending_acks[command_id]
    except Exception as exc:  # noqa: BLE001 - internal result must stay truthful
        logger.exception(
            "[media-control] %s failed user=%s channel=%s error=%s",
            action, user_id[:8], channel, type(exc).__name__,
        )
        return result(False, "error")
    ok, reason, was_playing = verdict
    if action == "pause" and ok and reason == "paused" and halted_item is not None:
        # The device paused the item this halt found on it (R6-9 residual 4):
        # a later `newer_playing` for it says paused, never "plays".
        state = _user_state(user_id)
        if state is not None and state.playing is halted_item:
            state.playing = replace(halted_item, paused=True)
    logger.info(
        "[media-control] %s user=%s command=%s reason=%s acked=%d silent=%d",
        action, user_id[:8], command_id[:8], reason, counts["acked"], counts["silent"],
    )
    return result(ok, reason, was_playing)


def _media_audience(ws_chat, user_id: str) -> list:
    """`ws_chat.media_command_audience`, or [] when it is unavailable (every
    recipient then counts as a socket that may be playing)."""

    try:
        audience = getattr(ws_chat, "media_command_audience", None)
        return list(audience(user_id)) if audience is not None else []
    except Exception:  # noqa: BLE001 - scoping must never break a stop
        logger.debug("[media-control] audience unavailable", exc_info=True)
        return []


def _note_missed(ws_chat, queues) -> None:
    try:
        missed = getattr(ws_chat, "_note_media_missed", None)
        if missed is not None:
            for queue in list(queues):
                missed(queue)
    except Exception:  # noqa: BLE001
        logger.debug("[media-control] missed-answer bookkeeping failed", exc_info=True)


def _video_id(session: Optional[RadioSession]) -> str:
    return str(getattr(session, "current_track_id", "") or "")


def _title(session: Optional[RadioSession]) -> str:
    track = getattr(session, "current_station_track", None)
    return str(getattr(track, "title", "") or "")


def _seed_video_id(session: Optional[RadioSession]) -> str:
    seed = getattr(session, "seed_track", None)
    return str(getattr(seed, "video_id", "") or "")


def _outcome(
    *,
    ok: bool,
    user_id: str,
    action: str,
    channel: str,
    reason: str,
    before: Optional[RadioSession] = None,
    after: Optional[RadioSession] = None,
    previous_video_id: Optional[str] = None,
) -> MediaControlOutcome:
    previous = _video_id(before) if previous_video_id is None else previous_video_id
    current = _video_id(after)
    return MediaControlOutcome(
        ok=ok,
        user_id=user_id,
        action=action,
        channel=channel,
        changed=bool(previous != current),
        reason=reason,
        previous_video_id=previous,
        video_id=current,
        title=_title(after),
    )


async def execute_media_control(
    user_id: str,
    action: str,
    channel: Optional[str] = None,
    *,
    ack_timeout_s: Optional[float] = None,
    before_order: object = None,
    media_scope: object = None,
    media_scope_started_ms: object = None,
) -> MediaControlOutcome:
    """Execute one radio navigation action and report its observed outcome.

    ``ok`` means the requested action completed and changed the active track.
    No session, a boundary no-op, and an exception are therefore all explicit
    false outcomes rather than optimistic acknowledgements.

    ``stop`` / ``pause`` are answered by the DEVICE: ``ok`` only on its ack,
    ``reason`` one of stopped | paused | nothing_playing | partially_confirmed |
    unacknowledged | delivery_failed | error. ``nothing_playing`` needs an
    answer from every reached socket that could be playing the channel;
    ``partially_confirmed`` (additive, ``ok`` false) is every answer idle while
    such a socket stayed silent. ``ack_timeout_s`` defaults to the endpoint
    budget minus a margin; once one device has answered without stopping
    anything, sockets not known to answer get ``_PEER_ACK_GRACE_S`` of it, not
    the rest.

    ``before_order`` / ``media_scope`` / ``media_scope_started_ms`` (stop/pause
    only, contract v0.3 §7): the caller's order for this halt. It supersedes
    the ordered plays it beats under the pairwise rule (same scope: order;
    different scopes, both stamped: stamp, first-seen on a tie; otherwise:
    arrival), in any scope; the device is stopped/paused unless the item last
    broadcast is one the caller asked for after it (or in its own turn) under
    the same rule — that NEWER item is left alone with ``reason``
    ``newer_playing`` (ok false, nothing sent to the device; ``paused`` true
    when that item is paused). Without an order and a scope the halt is
    unordered and behaves exactly as before.
    """

    normalized_action = str(action or "").strip().lower()
    normalized_channel = str(channel or "app").strip().lower() or "app"
    if normalized_action not in {"next", "previous", "stop", "pause"}:
        return _outcome(
            ok=False,
            user_id=user_id,
            action=normalized_action,
            channel=normalized_channel,
            reason="unsupported_action",
        )
    if normalized_action in _TRANSPORT_FRAMES:
        # Before the session check on purpose: a stop/pause does not need a
        # station, and "no station" must never read as "nothing to stop".
        return await _execute_transport(
            user_id,
            normalized_action,
            normalized_channel,
            _default_ack_timeout_s() if ack_timeout_s is None else float(ack_timeout_s),
            before_order=before_order,
            media_scope=media_scope,
            media_scope_started_ms=media_scope_started_ms,
        )
    if not RadioSessionManager.is_channel_allowed(normalized_channel):
        return _outcome(
            ok=False,
            user_id=user_id,
            action=normalized_action,
            channel=normalized_channel,
            reason="channel_not_allowed",
        )

    manager = get_radio_manager()
    before = manager.get(user_id, normalized_channel)
    if before is None or not before.enabled:
        return _outcome(
            ok=False,
            user_id=user_id,
            action=normalized_action,
            channel=normalized_channel,
            reason="no_active_session",
            before=before,
            after=before,
        )
    previous_video_id = _video_id(before)
    previous_seed_video_id = _seed_video_id(before)
    previous_epoch = int(getattr(before, "station_epoch", 0) or 0)

    # Import lazily: ws_chat owns the existing locks, refill rules, history,
    # variant resolution, and broadcasts.  Importing it at module load would
    # create a cycle through the API router.
    from app.api import ws_chat

    try:
        if normalized_action == "next":
            control_request = {
                "channel": normalized_channel,
                "reason": "user",
                "require_delivery": True,
            }
            handled = await ws_chat._handle_radio_skip_next(
                user_id, control_request,
            )
        else:
            control_request = {
                "channel": normalized_channel,
                "require_delivery": True,
            }
            handled = await ws_chat._handle_radio_skip_prev(
                user_id, control_request,
            )
    except Exception as exc:  # noqa: BLE001 - internal result must stay truthful
        after = manager.get(user_id, normalized_channel)
        logger.exception(
            "[media-control] action failed user=%s action=%s channel=%s error=%s",
            user_id[:8], normalized_action, normalized_channel, type(exc).__name__,
        )
        return _outcome(
            ok=False,
            user_id=user_id,
            action=normalized_action,
            channel=normalized_channel,
            reason="error",
            before=before,
            after=after,
            previous_video_id=previous_video_id,
        )

    after = manager.get(user_id, normalized_channel)
    changed = previous_video_id != _video_id(after)
    seed_changed = previous_seed_video_id != _seed_video_id(after)
    epoch_changed = previous_epoch != int(getattr(after, "station_epoch", 0) or 0)
    detail = control_request.get("_media_control_result") or {}
    handler_reason = str(detail.get("reason") or "")
    if handler_reason == "no_active_session":
        # The service already proved an active session before dispatch.  If
        # the locked handler no longer sees it, an OFF/reseed won the race;
        # that is supersession, not a claim that no session existed when the
        # command was accepted.
        reason = "superseded"
    elif handler_reason in {
        "superseded",
        "delivery_failed",
        "exhausted",
        "channel_not_allowed",
    }:
        reason = handler_reason
    elif handler_reason == "unchanged":
        reason = "unchanged"
    elif seed_changed or (epoch_changed and not handler_reason):
        reason = "superseded"
    elif after is None or not after.enabled:
        # The operation began with an active session. Losing/exhausting it
        # during dispatch is an execution error, not "there was no session".
        reason = "error"
    elif not handled and changed:
        # A changed cursor with a false handler result is never success. It is
        # either a delivery failure after mutation or a concurrent reseed that
        # superseded this command while a refill was awaiting the network.
        reason = (
            "superseded"
            if seed_changed
            else "error"
        )
    elif not handled or not changed:
        reason = "unchanged"
    else:
        reason = "advanced" if normalized_action == "next" else "rewound"
    return _outcome(
        ok=bool(
            handled and changed and not seed_changed
            and handler_reason not in {
                "superseded", "delivery_failed", "exhausted",
                "channel_not_allowed", "no_active_session", "unchanged",
            }
            and after is not None and after.enabled
        ),
        user_id=user_id,
        action=normalized_action,
        channel=normalized_channel,
        reason=reason,
        before=before,
        after=after,
        previous_video_id=previous_video_id,
    )
