"""R43 repair — the two holes in the delivery writers.

Finding 16. `_check_addressed_to_the_user` fell off the end with a bare
`return` for `teams_chat`, `notion_page` and `calendar_hold`, so three
of the five connector channels were checked for nothing but a non-empty
target. A grant pinned to a group chat, a team page or a shared calendar
is a perfectly LEGAL grant for a write STEP — that is the whole reason
§1.3 exists — and those three would have posted the user's ranked brief,
their mail subjects and their board into it, under a sheet that says
"Every channel here is you".

Finding 17. A WhatsApp / Telegram delivery that failed `return`ed a
failed dict instead of raising, and `deliver_brief` only appends the
thread turn on `DeliveryRefused` — so the brief silently stopped
arriving and the only trace was a job-config blob no surface renders.
The module's own rule 3 is that a channel which cannot be written to
fails VISIBLY.

Everything here drives the real functions; nothing re-implements them.
"""

from __future__ import annotations

import pytest

from app.agent.automations import catalog, deliver


# ── finding 16: the ownership check is TOTAL ─────────────────────────


def _check(channel_id, target, account, owned=None):
    spec = deliver._CONNECTOR_CHANNELS.get(channel_id) or {}
    deliver._check_addressed_to_the_user(
        channel_id, spec.get("connector_id") or "", target, account, owned)


_GROUP_CHAT = "19:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee@thread.v2"
_SELF_CHAT = "19:11111111_11111111@unq.gbl.spaces"


@pytest.mark.parametrize("channel_id,target,account,owned,ok", [
    # Teams. The user's own chat is the ONLY one that reaches them and
    # nobody else, and it has to be PROVEN — a chat id is not an
    # identity, so the grant cannot be its own evidence.
    ("teams_chat", _SELF_CHAT, "", {_SELF_CHAT}, True),
    ("teams_chat", _GROUP_CHAT, "", {_SELF_CHAT}, False),
    # The exact finding-16 scenario: a legitimate write-step grant
    # pinned to a chat with three colleagues in it.
    ("teams_chat", _GROUP_CHAT, "me@acme.com", None, False),
    # An owner nobody could enumerate is a refusal, never a pass.
    ("teams_chat", _SELF_CHAT, "me@acme.com", set(), False),

    # Notion exposes no sharing state at all, so no target can be
    # proven and the channel refuses rather than guessing.
    ("notion_page", "page-1", "me@acme.com", None, False),
    ("notion_page", "9d2f0a1b4c6e4f0a8b3c5d7e9f0a1b2c", "", None, False),

    # Calendar. `primary` is the literal alias for the token owner's own
    # calendar; so is the account's own address. A shared calendar is
    # somebody else's room and a hold sitting in it is readable by
    # everyone who can read the calendar.
    ("calendar_hold", "primary", "", None, True),
    ("calendar_hold", "PRIMARY", "", None, True),
    ("calendar_hold", "me@acme.com", "me@acme.com", None, True),
    ("calendar_hold", "ME@ACME.COM", "me@acme.com", None, True),
    ("calendar_hold", "team@group.calendar.google.com", "me@acme.com",
     None, False),
    ("calendar_hold", "dana@acme.com", "me@acme.com", None, False),
    ("calendar_hold", "someone@acme.com", "", None, False),

    # The two that already had branches keep them (regression fence:
    # the tail rewrite must not have eaten either `return`).
    ("gmail_draft", "me@acme.com", "me@acme.com", None, True),
    ("slack_dm", "D0FOUNDER", "U0ME", None, True),
    ("slack_dm", "C0ALL", "U0ME", None, False),
])
def test_no_delivery_channel_reaches_anyone_but_the_user(
        channel_id, target, account, owned, ok):
    if ok:
        _check(channel_id, target, account, owned)
    else:
        with pytest.raises(deliver.DeliveryRefused):
            _check(channel_id, target, account, owned)


def test_the_ownership_check_has_a_branch_for_every_shipped_channel():
    """Not "does it raise" — does it DECIDE. A channel with no branch
    used to fall off the end and pass; the check must now answer for
    every id in the table, and the answer must not be the catch-all."""
    for cid in deliver._CONNECTOR_CHANNELS:
        try:
            _check(cid, "some-target-nobody-can-prove", "", set())
        except deliver.DeliveryRefused as e:
            assert e.reason_code != "unverifiable_channel" or (
                cid in deliver.UNVERIFIABLE_CHANNELS), (
                f"{cid} reached the no-branch catch-all")
        else:  # pragma: no cover — a pass here is the finding-16 bug
            pytest.fail(f"{cid} passed an unprovable target")


def test_a_channel_with_no_branch_refuses_instead_of_falling_through():
    """The closing raise, which is what makes the function total. A
    tenth channel added to the catalogue and forgotten here must be
    unable to deliver, rather than able to deliver anywhere."""
    with pytest.raises(deliver.DeliveryRefused) as e:
        deliver._check_addressed_to_the_user(
            "some_future_channel", "future", "anything", "me@acme.com")
    assert e.value.reason_code == "unverifiable_channel"
    assert e.value.sentence


def test_the_unverifiable_table_says_why_in_words():
    """§0.2 — the option is not offered AND the UI says why.
    `workflow._channel_state` prints these verbatim, and the ledger's
    grammar drops any served string carrying a raw tool id."""
    assert deliver.UNVERIFIABLE_CHANNELS, "the table is the reason field"
    for cid, sentence in deliver.UNVERIFIABLE_CHANNELS.items():
        assert cid in deliver._CONNECTOR_CHANNELS, cid
        assert catalog.channel(cid), f"{cid} is not a catalogue channel"
        assert sentence and sentence[0].isupper(), sentence
        assert "__" not in sentence, "a raw tool id would be dropped"
        assert not sentence.endswith("."), "the string table sets its own stop"


@pytest.mark.asyncio
async def test_only_the_users_own_teams_chat_counts_as_owned(monkeypatch):
    """`_owned_chats` reuses the teams provider's own closed
    self-identification (`_identify_self`) rather than re-deriving it,
    and keeps only chats whose member set is exactly {self}. A oneOnOne
    with a colleague reaches the user AND the colleague."""
    from app.connectors.teams import provider as _teams

    chats = [
        {"id": _SELF_CHAT, "chatType": "oneOnOne",
         "members": [{"userId": "me"}]},
        {"id": "19:me_dana@unq.gbl.spaces", "chatType": "oneOnOne",
         "members": [{"userId": "me"}, {"userId": "dana"}]},
        {"id": _GROUP_CHAT, "chatType": "group",
         "members": [{"userId": "me"}, {"userId": "dana"},
                     {"userId": "sam"}]},
    ]

    async def _token(user_id):
        return "tok"

    async def _graph(method, url, **kw):
        return {"value": chats}

    monkeypatch.setattr(_teams, "_resolve_token", _token)
    monkeypatch.setattr(_teams, "_graph", _graph)
    assert await deliver._owned_chats("u") == {_SELF_CHAT}


@pytest.mark.asyncio
async def test_an_unreachable_teams_account_owns_nothing(monkeypatch):
    """Nothing on the delivery path may raise, and an unprovable owner
    is a REFUSAL rather than a guess — so the failure has to come back
    as an empty set, which every caller treats as "no target passes"."""
    from app.connectors.teams import provider as _teams

    async def _boom(user_id):
        raise RuntimeError("No active Teams identity")

    monkeypatch.setattr(_teams, "_resolve_token", _boom)
    assert await deliver._owned_chats("u") == set()
    with pytest.raises(deliver.DeliveryRefused):
        _check("teams_chat", _SELF_CHAT, "", set())


@pytest.mark.asyncio
async def test_only_teams_asks_for_an_enumeration(monkeypatch):
    """Every other channel proves ownership by comparing one identity,
    which is free. `_owned_targets` must not fan a provider call out
    across channels that do not need one."""
    called: list = []

    async def _chats(user_id):
        called.append(user_id)
        return {"x"}

    monkeypatch.setattr(deliver, "_owned_chats", _chats)
    for cid in deliver._CONNECTOR_CHANNELS:
        got = await deliver._owned_targets("u", cid)
        assert got == ({"x"} if cid == "teams_chat" else set()), cid
    assert called == ["u"]


@pytest.mark.asyncio
async def test_a_teams_delivery_stages_nothing_when_the_chat_is_not_yours(
        monkeypatch):
    """The wiring, not just the predicate: `_deliver_one` must ASK for
    the enumeration and hand it to the check, and it must refuse before
    the outbox row is built. `db` is None here on purpose — reaching it
    would be the bug."""
    async def _grant(user_id, automation, connector_id, tool):
        return {"id": "g", "target": {"id": _GROUP_CHAT, "label": "team"}}

    async def _account(user_id, connector_id):
        return "me@acme.com"

    async def _owned(user_id, channel_id):
        return {_SELF_CHAT}

    monkeypatch.setattr(deliver, "_grant_for", _grant)
    monkeypatch.setattr(deliver, "_account_for", _account)
    monkeypatch.setattr(deliver, "_owned_targets", _owned)

    class _A:
        id = "a1"
        user_id = "u1"
        name = "Morning brief"

    with pytest.raises(deliver.DeliveryRefused) as e:
        await deliver._deliver_one(
            None, channel_id="teams_chat", automation=_A(), job_id="j",
            thread=None, brief=_brief(), now=_now(), idem_prefix="t")
    assert e.value.reason_code == "not_the_user"
    assert "__" not in e.value.sentence


# ── finding 17: an own-channel failure is VISIBLE ────────────────────


_GROUPS = [
    {"rank": 1, "label": "DO FIRST · BLOCKS OTHERS", "tone": "danger",
     "rows": [{"text": "Dana needs an owner for the retry flag",
               "sub": "It blocks the client fix going out tonight.",
               "tag": "P1", "item_refs": ["it_1"]}],
     "items": [{"id": "it_1", "who": "Dana Cole",
                "title": "Anyone own the retry flag?", "sub": "",
                "why": "", "at": "2026-08-31T22:40:00Z",
                "source": "slack", "where": "#platform", "hot": True}],
     "empty_reason": None},
]


def _now():
    from datetime import datetime, timezone
    return datetime(2026, 9, 1, 8, 0, tzinfo=timezone.utc)


def _brief():
    from app.agent.automations.brief_render import brief_render
    return brief_render(_GROUPS, "ranked", title="T", slug="brief")


class _Auto:
    id = "a1"
    user_id = "u1"
    name = "Morning brief"


def _capture_turns(monkeypatch) -> list:
    seen: list = []

    async def _turn(db, *, automation, job_id, thread, channel_id, refusal):
        seen.append({"channel": channel_id, "reason": refusal.reason_code,
                     "sentence": refusal.sentence})

    async def _rec(db, *, automation, job_id, thread, results, plan):
        return None

    monkeypatch.setattr(deliver, "_append_refused_turn", _turn)
    monkeypatch.setattr(deliver, "_record", _rec)
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("entry,reason", [
    ({"status": "skipped", "reason": "no_recipient"}, "no_recipient"),
    ({"status": "skipped", "reason": "no_adapter"}, "no_adapter"),
    ({"status": "failed", "error_class": "TimeoutError",
      "error_detail": "read timeout"}, "TimeoutError"),
    ({}, "no_adapter"),
])
async def test_a_dropped_whatsapp_session_reaches_the_thread(
        monkeypatch, entry, reason):
    """Rule 3, and the half that was missing. WhatsApp and Telegram skip
    the outbox — which is what appends the failed-write turn for every
    OTHER channel — so this file has to raise, or the brief silently
    stops arriving."""
    async def _detailed(*, user_id, delivery_channels, routine_name,
                        content, db_session_maker):
        return {delivery_channels[0]: dict(entry)}

    monkeypatch.setattr(
        "app.agent.routines.channel_dispatcher"
        ".deliver_to_extra_channels_detailed", _detailed)
    seen = _capture_turns(monkeypatch)

    out = await deliver.deliver_brief(
        None, automation=_Auto(), job_id="j", thread=object(),
        groups=_GROUPS, title="T",
        delivery={"channels": ["whatsapp"], "format": "ranked",
                  "cadence": "run"},
        idem_prefix="t")

    assert out["whatsapp"] == {"status": "failed", "reason": reason}
    assert [s["channel"] for s in seen] == ["whatsapp"], (
        "a failed own-channel delivery left no record in the thread")
    assert seen[0]["sentence"], "the turn needs words, not a reason token"
    assert "__" not in seen[0]["sentence"]


@pytest.mark.asyncio
async def test_a_delivered_own_channel_says_nothing(monkeypatch):
    """The other side of the same branch: success must not append a
    failure turn, and the return shape the job config records is
    unchanged."""
    async def _detailed(*, user_id, delivery_channels, routine_name,
                        content, db_session_maker):
        return {delivery_channels[0]: {"status": "delivered"}}

    monkeypatch.setattr(
        "app.agent.routines.channel_dispatcher"
        ".deliver_to_extra_channels_detailed", _detailed)
    seen = _capture_turns(monkeypatch)

    out = await deliver.deliver_brief(
        None, automation=_Auto(), job_id="j", thread=object(),
        groups=_GROUPS, title="T",
        delivery={"channels": ["app", "telegram"], "format": "ranked",
                  "cadence": "run"},
        idem_prefix="t")
    assert out["telegram"] == {"status": "delivered", "reason": None}
    assert out["app"]["status"] == "skipped"
    assert seen == []


@pytest.mark.asyncio
async def test_an_unexpected_delivery_fault_is_recorded_too(monkeypatch):
    """The generic arm. Nothing after `flush_row_when_due` can raise, so
    an exception here means the outbox never staged the row and never
    appended its own turn — this is the user's only record."""
    async def _boom(*a, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(deliver, "_deliver_one", _boom)
    seen = _capture_turns(monkeypatch)

    out = await deliver.deliver_brief(
        None, automation=_Auto(), job_id="j", thread=object(),
        groups=_GROUPS, title="T",
        delivery={"channels": ["slack_dm"], "format": "ranked",
                  "cadence": "run"},
        idem_prefix="t")
    assert out["slack_dm"] == {"status": "failed", "reason": "unknown_error"}
    assert [s["channel"] for s in seen] == ["slack_dm"]


@pytest.mark.asyncio
async def test_a_brief_that_could_not_be_made_is_not_a_silent_no_show(
        monkeypatch):
    """`PdfUnavailable` failed every channel and told nobody. `app` is
    excluded on purpose — the thread already holds the brief, only the
    file is missing, so "it did not reach this chat" would be false."""
    from app.agent.automations import brief_render as _br

    def _boom(groups, format_id, *, title="", slug="brief"):
        raise _br.PdfUnavailable("no renderer")

    monkeypatch.setattr(deliver, "brief_render", _boom)
    seen = _capture_turns(monkeypatch)

    out = await deliver.deliver_brief(
        None, automation=_Auto(), job_id="j", thread=object(),
        groups=_GROUPS, title="T",
        delivery={"channels": ["app", "whatsapp"], "format": "pdf",
                  "cadence": "run"},
        idem_prefix="t")
    assert out["whatsapp"]["reason"] == "format_unavailable"
    assert [s["channel"] for s in seen] == ["whatsapp"]


def test_every_own_channel_sentence_is_something_to_read():
    """`channel_dispatcher` answers in tokens (`no_recipient`) and
    exception class names. The ledger's grammar drops a turn carrying a
    raw tool id, and a person cannot read either one."""
    for reason, sentence in deliver._OWN_CHANNEL_SENTENCES.items():
        assert reason.islower() and " " not in reason, reason
        assert sentence and sentence[0].isupper(), sentence
        assert "__" not in sentence and "_" not in sentence, sentence


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_id,reason,sentence", [
    # No connector behind it: every string the health table composes
    # would be about an account that does not exist.
    ("whatsapp", "no_recipient", "It has no number for you"),
    ("telegram", "no_adapter", "That channel is not connected"),
    # And the connector-backed refusals keep working.
    ("teams_chat", "not_the_user", "A Teams delivery goes to your own chat"),
    ("notion_page", "unverifiable_channel", "Notion cannot say who else"),
])
async def test_the_refusal_turn_survives_a_channel_with_no_account(
        monkeypatch, channel_id, reason, sentence):
    """The whole body of `_append_refused_turn` is swallowed by a
    `try/except` — a crash inside it is SILENCE, which is the very bug
    finding 17 is about. So it is driven for real here, with only the
    ledger write stubbed."""
    from app.agent.automations import ledger as _ledger

    wrote: list = []

    async def _append(db, *, user_id, thread, run_id, kind, payload):
        wrote.append(payload)

    monkeypatch.setattr(_ledger, "append_turn", _append)
    await deliver._append_refused_turn(
        None, automation=_Auto(), job_id="j", thread=object(),
        channel_id=channel_id,
        refusal=deliver.DeliveryRefused(reason, "detail", sentence=sentence))

    assert len(wrote) == 1, "a failed delivery left no record in the thread"
    p = wrote[0]
    assert p["ok"] is False and p["tool_kind"] == "write"
    assert p["line"] and "__" not in p["line"]
    assert p["reason_code"]
    assert sentence in [s["text"] for s in p["steps"]]
    for s in p["steps"]:
        assert "__" not in s["text"], s
    # A "Retry" under "that chat is not yours" offers a fix that cannot
    # work; the button and the sentence are dropped together.
    if channel_id in ("whatsapp", "telegram"):
        assert "fix" not in p
