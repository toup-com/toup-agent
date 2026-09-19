"""The fleet page must follow the CONDITION, and one replica must own the tick.

WHY THIS EXISTS — the ToupInfraAlertsBot feed, 2026-09-16:

  * 04:58Z and 07:31Z each carried the SAME fleet warning TWICE, ~11 s apart.
    `rollout_reconciler_loop` was created unconditionally per replica
    (`app/main.py`) and Railway runs two; the bridge access log shows two
    platform processes polling `/v1/pool/health` every 30 s, 11 s apart. Seven
    other periodic loops on this service hold an infra lease. This one held
    nothing — and the three exclusions it does have (`_resume_inflight`,
    `_rollout_creation_lock`, `_FLEET_STATE`) are all process-local.
  * The gate was a single per-process timestamp, consumed BEFORE the await and
    regardless of what Telegram answered. Two consequences, both of them
    live: a 429 (the canonical result of two replicas posting to one chat
    0.3 s apart) suppressed the warning for the whole 6 h interval against a
    message nobody received; and the gate was content-blind, so an escalation
    from "1 slot behind" to "83 of 84 slots behind" was swallowed inside one
    window — eight lines higher, the same function change-detects its LOG
    line.

So: one runner, a gate keyed on a coarse GRADE of the fault, and state
written only after a confirmed 2xx.

The grade is coarse on purpose, and that is the half a first pass got wrong.
Keying on the raw counts pages on any movement INCLUDING improvement, and a
manual fleet walk converges one slot every ~2 minutes for hours (round 46: 84
slots) — so a count-keyed gate would have paged ~84 times for the very episode
above. Only a WORSENING is news; the exact live counts still go in the body.

The state is per-process (R47 forbids a new table), so a standing fault costs
one page per platform-api process — bounded by the deploy rate, and stated in
the report.

Run:
  cd backend && RUN_MODE=platform PYTHONPATH=. python -m pytest -q \
      tests/test_rollout_fleet_alerting.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.services.rollout_service as RS  # noqa: E402


CURRENT = "ghcr.io/toup-com/toup-agent:7edaed3ab644"
NEXT = "ghcr.io/toup-com/toup-agent:166b835e5ec4"
FLEET = 84
TICK = 30.0


def _fleet(behind: int, *, generic: int = 0, auto: bool = False,
           tag: str = CURRENT, total: int = FLEET) -> dict:
    """The 2026-09-16 shape: auto-upgrade paused with slots behind, which
    warns immediately (nothing will converge them) — no 2 h split grace to
    wait out, so a count/band change is visible on the next tick."""
    return {
        "images": {tag: total - behind, NEXT: behind},
        "current_image_tag": tag,
        "assigned_total": total,
        "assigned_on_current": total - behind,
        "assigned_on_other": behind,
        "generic_on_other": generic,
        "auto_upgrade_assigned": auto,
    }


CONVERGED = dict(_fleet(0, auto=True), images={CURRENT: FLEET})


def _seed_split_grace(now: float) -> None:
    """Age the split clock past `rollout_fleet_split_alert_after_s`.

    The `split_stalled` condition only arms once the split has STOOD for that
    long (a split is normal while a rollout converges), so a test that wants
    the auto-upgrade-ON arm warning from its first tick has to pre-date the
    split rather than tick for two hours. `_FLEET_STATE["split_since"]` is the
    clock `fleet_status` keeps, and the autouse fixture clears it again.
    """
    RS._FLEET_STATE["split_since"] = (
        now - float(RS.settings.rollout_fleet_split_alert_after_s) - 60.0
    )


@pytest.fixture(autouse=True)
def _clean_state():
    RS._reset_fleet_watch_state()
    yield
    RS._reset_fleet_watch_state()


@pytest.fixture
def configured():
    """A deployment that HAS a Telegram target.

    Without this, `_fleet_page` is right to treat a False verdict as "nothing
    to deliver" rather than "refused" — so every test about a REFUSAL has to
    say that alerting is configured, or it is testing the other branch.
    """
    with patch.object(RS.settings, "infra_alert_telegram_token", "tok"), \
         patch.object(RS.settings, "infra_alert_telegram_chat_id", "-1001"):
        yield


def _health(fleet):
    return {"ok": True, "current_image_tag": fleet["current_image_tag"], "fleet": fleet}


class _Feed:
    """A mutable `_bridge_get` — the fleet the bridge reports, tick by tick."""

    def __init__(self, fleet):
        self.fleet = fleet

    async def __call__(self, path, *, timeout_s=0):
        return (_health(self.fleet), "ok")


class _Pager:
    """A `_send_telegram` that records, and answers a delivery verdict."""

    def __init__(self, delivered: bool = True):
        self.delivered = delivered
        self.sent: list[tuple[str, str]] = []

    async def __call__(self, level, message):
        self.sent.append((level, message))
        return self.delivered

    @property
    def levels(self):
        return [lvl for lvl, _ in self.sent]


# ─── the grade itself ─────────────────────────────────────────────


class TestSeverityGrade:
    def test_the_band_is_a_fraction_of_the_fleet_not_a_count(self):
        """Coarse on purpose: bands are 0 / (0,10%] / (10%,50%] / (50%,100%]
        of the assigned fleet, so one converging slot is invisible to the
        gate and a jump across half the fleet is not."""
        assert RS._fleet_severity_band(0, 84) == 0
        assert RS._fleet_severity_band(1, 84) == 1
        assert RS._fleet_severity_band(8, 84) == 1     # 9.5%
        assert RS._fleet_severity_band(9, 84) == 2     # 10.7%
        assert RS._fleet_severity_band(42, 84) == 2    # exactly 50%
        assert RS._fleet_severity_band(43, 84) == 3
        assert RS._fleet_severity_band(84, 84) == 3

    def test_the_grade_reads_assigned_total_or_derives_it(self):
        """And the derived denominator is what covers a bridge whose `fleet`
        block predates `assigned_total` — including one that reports no
        on-current count either, which derives total == behind, i.e. the top
        band through the NORMAL path. `_fleet_severity_band` therefore needs no
        no-denominator special case, and has none."""
        g = RS._fleet_alert_grade(_fleet(1), ["auto_upgrade_off"])
        assert g["band"] == 1 and g["tag"] == CURRENT and g["generic"] is False
        # Same fleet, no `assigned_total` key: on_current + behind is that
        # number by construction (bridge/pool_addon.py::_fleet_snapshot).
        no_total = {k: v for k, v in _fleet(1).items() if k != "assigned_total"}
        assert RS._fleet_alert_grade(no_total, ["auto_upgrade_off"])["band"] == 1
        # Neither count present: 5 behind out of a fleet of 5 knowable slots.
        blind = {"current_image_tag": CURRENT, "assigned_on_other": 5,
                 "auto_upgrade_assigned": False}
        assert RS._fleet_alert_grade(blind, ["auto_upgrade_off"])["band"] == 3

    def test_only_the_worsening_directions_are_news(self):
        low = RS._fleet_alert_grade(_fleet(1), ["auto_upgrade_off"])
        high = RS._fleet_alert_grade(_fleet(83), ["auto_upgrade_off"])
        assert RS._fleet_alert_worsened(None, low) is True
        assert RS._fleet_alert_worsened(low, high) is True      # band up
        assert RS._fleet_alert_worsened(high, low) is False     # band down
        assert RS._fleet_alert_worsened(high, high) is False
        newtag = RS._fleet_alert_grade(_fleet(83, tag=NEXT), ["auto_upgrade_off"])
        assert RS._fleet_alert_worsened(high, newtag) is True
        newkind = RS._fleet_alert_grade(_fleet(83), ["split_stalled"])
        assert RS._fleet_alert_worsened(high, newkind) is True
        gen = RS._fleet_alert_grade(_fleet(83, generic=10), ["auto_upgrade_off"])
        assert RS._fleet_alert_worsened(high, gen) is True      # generics joined
        assert RS._fleet_alert_worsened(gen, high) is False     # generics drained


# ─── the page follows the condition ───────────────────────────────


class TestWorseningGate:
    @pytest.mark.asyncio
    async def test_an_escalation_pages_inside_the_window(self):
        """THE defect. 1 of 84 behind then 83 of 84 is not a repeat of the
        same news; the bare timestamp suppressed it for 6 h."""
        feed, pager = _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            await RS._fleet_watch_once(now=1_000.0 + TICK)      # unchanged
            feed.fleet = _fleet(83)
            await RS._fleet_watch_once(now=1_000.0 + 2 * TICK)  # escalation
        assert len(pager.sent) == 2, pager.sent
        assert "1 slot(s)" in pager.sent[0][1]
        assert "83 slot(s)" in pager.sent[1][1]

    @pytest.mark.parametrize("auto", [False, True])
    @pytest.mark.asyncio
    async def test_a_convergence_walk_pages_nothing_beyond_the_first(self, auto):
        """The noise regression a count-keyed gate reintroduces. A manual
        fleet walk drops the count by one every ~2 min for hours (round 46:
        84 slots) — 41 steps here, ALL inside the top band.

        BOTH arms, because they page different text and only one of them is
        obviously safe: `auto_upgrade_off` (auto=False) names a count, while
        `split_stalled` (auto=True) embeds the elapsed minutes and therefore
        changes on EVERY tick. A gate keyed on the warning text would be
        silent on the first arm and page 41 times on the second — and the
        2026-09-16 episode this file cites was partly the second.
        """
        t0 = 1_000_000.0
        feed, pager = _Feed(_fleet(FLEET, auto=auto)), _Pager()
        _seed_split_grace(t0)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i, behind in enumerate(range(FLEET, 42, -1)):
                feed.fleet = _fleet(behind, auto=auto)
                await RS._fleet_watch_once(now=t0 + 120.0 * i)
        assert len(pager.sent) == 1, f"{len(pager.sent)} pages for one episode"
        assert ("84 assigned" if auto else "84 slot(s)") in pager.sent[0][1]

    @pytest.mark.parametrize("auto", [False, True])
    @pytest.mark.asyncio
    async def test_a_fall_across_bands_is_still_not_news(self, auto):
        """43/84 (top band) down to 8/84 (lowest) — better, so silent. The
        operator already knows about the worse reading; the recovery notice
        reports the end of the episode.

        Both arms again, and the second tick is four ticks later rather than
        one so that the `split_stalled` text differs in its ELAPSED MINUTES as
        well as its slot count — which is what makes this a pin on the gate
        reading the grade rather than the words."""
        t0 = 1_000_000.0
        feed, pager = _Feed(_fleet(43, auto=auto)), _Pager()
        _seed_split_grace(t0)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=t0)
            feed.fleet = _fleet(8, auto=auto)
            await RS._fleet_watch_once(now=t0 + 4 * TICK)
        assert len(pager.sent) == 1, pager.sent

    @pytest.mark.asyncio
    async def test_a_rise_inside_one_band_is_not_news_either(self):
        """The band is the granularity an operator acts on: 1 → 8 of 84 is
        still "a few slots", and pinning it keeps the gate coarse in BOTH
        directions rather than only downwards."""
        feed, pager = _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            feed.fleet = _fleet(8)
            await RS._fleet_watch_once(now=1_000.0 + TICK)
            feed.fleet = _fleet(9)                              # 10.7% — band 2
            await RS._fleet_watch_once(now=1_000.0 + 2 * TICK)
        assert len(pager.sent) == 2, pager.sent
        assert "9 slot(s)" in pager.sent[1][1]

    @pytest.mark.asyncio
    async def test_the_walk_to_zero_closes_with_one_notice_then_a_fresh_fault_pages(self):
        """The whole episode, end to end: one page, silence while it
        converges, one closing notice, and the next fault is not held behind
        the old one's window."""
        feed, pager = _Feed(_fleet(FLEET)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i, behind in enumerate(range(FLEET, -1, -1)):
                feed.fleet = _fleet(behind) if behind else CONVERGED
                await RS._fleet_watch_once(now=1_000.0 + 120.0 * i)
            feed.fleet = _fleet(FLEET)
            await RS._fleet_watch_once(now=1_000.0 + 120.0 * (FLEET + 2))
        assert pager.levels == ["warning", "info", "warning"], pager.sent
        assert "converged" in pager.sent[1][1].lower()

    @pytest.mark.asyncio
    async def test_generics_joining_the_split_pages_and_draining_does_not(self):
        """Generic slots have no total in the bridge's fleet block
        (`_fleet_snapshot` has `assigned_total` and no generic one), so they
        are graded as a boolean. "Are there any" is the actionable part."""
        feed, pager = _Feed(_fleet(83)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            feed.fleet = _fleet(83, generic=10)
            await RS._fleet_watch_once(now=1_000.0 + TICK)
            for i in range(10, -1, -1):                 # the generics drain
                feed.fleet = _fleet(83, generic=i)
                await RS._fleet_watch_once(now=1_000.0 + TICK * (3 + i))
        assert len(pager.sent) == 2, pager.sent

    @pytest.mark.asyncio
    async def test_an_unchanged_fault_is_suppressed_for_the_whole_interval(self):
        feed, pager = _Feed(_fleet(1)), _Pager()
        interval = float(RS.settings.rollout_fleet_alert_interval_s)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i in range(int(interval // TICK)):
                await RS._fleet_watch_once(now=1_000.0 + TICK * i)
        assert len(pager.sent) == 1, f"{len(pager.sent)} pages in one window"

    @pytest.mark.asyncio
    async def test_an_unchanged_fault_pages_again_after_the_interval(self):
        """Suppression is a window, not a mute — a fault still standing six
        hours later is worth saying again."""
        feed, pager = _Feed(_fleet(1)), _Pager()
        interval = float(RS.settings.rollout_fleet_alert_interval_s)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            await RS._fleet_watch_once(now=1_000.0 + interval - 1)
            await RS._fleet_watch_once(now=1_000.0 + interval)
        assert len(pager.sent) == 2, pager.sent

    @pytest.mark.asyncio
    async def test_an_improvement_still_pages_once_the_interval_elapses(self):
        """The suppression of an improvement is a WINDOW too, so the grade the
        operator knows decays with it: six hours later the re-page carries the
        current, lower count."""
        feed, pager = _Feed(_fleet(FLEET)), _Pager()
        interval = float(RS.settings.rollout_fleet_alert_interval_s)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            feed.fleet = _fleet(1)
            await RS._fleet_watch_once(now=1_000.0 + TICK)
            await RS._fleet_watch_once(now=1_000.0 + interval)
        assert len(pager.sent) == 2, pager.sent
        assert "1 slot(s)" in pager.sent[1][1]

    @pytest.mark.asyncio
    async def test_a_new_target_tag_is_a_new_fault(self):
        """The same count behind a DIFFERENT tag is a different rollout stuck,
        so the grade carries the tag."""
        feed, pager = _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            feed.fleet = _fleet(1, tag=NEXT)
            await RS._fleet_watch_once(now=1_000.0 + TICK)
        assert len(pager.sent) == 2, pager.sent

    @pytest.mark.asyncio
    async def test_the_page_body_carries_the_exact_live_counts(self):
        """The GATE is coarse; the message is not. An operator acting on a
        page needs the number, not the band."""
        feed, pager = _Feed(_fleet(37)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
        assert "37 slot(s) behind" in pager.sent[0][1], pager.sent

    @pytest.mark.asyncio
    async def test_convergence_closes_the_edge_exactly_once(self):
        """bridge_supervisor's alarm/recovery shape: one notice per episode,
        and none at all if the operator was never paged."""
        feed, pager = _Feed(CONVERGED), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i in range(3):                      # never faulted: silence
                await RS._fleet_watch_once(now=1_000.0 + TICK * i)
            assert pager.sent == []
            feed.fleet = _fleet(1)
            await RS._fleet_watch_once(now=2_000.0)
            feed.fleet = CONVERGED
            for i in range(3):                      # one recovery, not three
                await RS._fleet_watch_once(now=2_100.0 + TICK * i)
        assert pager.levels == ["warning", "info"], pager.sent
        assert "converged" in pager.sent[1][1].lower()

    @pytest.mark.asyncio
    async def test_the_recovery_notice_never_claims_a_split_fleet_converged(self):
        """"No warnings" is not "converged". Re-enabling auto-upgrade clears
        the fault with every slot still behind, and the operator who was told
        "nothing will converge them" would read "converged" as done."""
        feed, pager = _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            feed.fleet = _fleet(1, auto=True)
            await RS._fleet_watch_once(now=1_000.0 + TICK)
        assert pager.levels == ["warning", "info"], pager.sent
        notice = pager.sent[1][1]
        assert "converged" not in notice.lower(), notice
        assert "1 slot(s) still off" in notice, notice


# ─── a refused send must not consume the window ───────────────────


class TestDeliveryVerdict:
    @pytest.mark.asyncio
    async def test_send_telegram_returns_the_delivery_verdict(self):
        """`send_infra_alert` is 2xx-confirmed and this wrapper discarded the
        answer, which threw away the one property alerting.py was rewritten
        to have (L3-7)."""
        with patch("app.services.alerting.send_infra_alert", AsyncMock(return_value=True)):
            assert await RS._send_telegram("warning", "x") is True
        with patch("app.services.alerting.send_infra_alert", AsyncMock(return_value=False)):
            assert await RS._send_telegram("warning", "x") is False

    @pytest.mark.asyncio
    async def test_a_refused_send_does_not_consume_the_window(self, configured):
        """A 429 used to buy the fault six hours of silence. It must be
        retried instead — and it matters MORE now the loop is leased, because
        there is no second replica whose duplicate page covered the loss."""
        feed, pager = _Feed(_fleet(1)), _Pager(delivered=False)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            await RS._fleet_watch_once(now=1_000.0 + RS._FLEET_ALERT_RETRY_S)
            pager.delivered = True
            await RS._fleet_watch_once(now=1_000.0 + 2 * RS._FLEET_ALERT_RETRY_S)
            assert len(pager.sent) == 3, pager.sent
            # …and once it lands, the window IS consumed.
            await RS._fleet_watch_once(now=1_000.0 + 2 * RS._FLEET_ALERT_RETRY_S + TICK)
        assert len(pager.sent) == 3, pager.sent

    @pytest.mark.asyncio
    async def test_a_refused_send_is_not_retried_on_every_tick(self, configured):
        """Not consuming the window is not licence to POST every 30 s at an
        API that just said no."""
        feed, pager = _Feed(_fleet(1)), _Pager(delivered=False)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i in range(int(RS._FLEET_ALERT_RETRY_S // TICK)):
                await RS._fleet_watch_once(now=1_000.0 + TICK * i)
        assert len(pager.sent) == 1, pager.sent

    @pytest.mark.asyncio
    async def test_a_refused_send_never_delays_a_worsening(self, configured):
        """The retry spacing is graded too, so worse news is not held behind a
        failed attempt to report better news."""
        feed, pager = _Feed(_fleet(1)), _Pager(delivered=False)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            feed.fleet = _fleet(83)
            await RS._fleet_watch_once(now=1_000.0 + TICK)
        assert len(pager.sent) == 2, pager.sent

    @pytest.mark.asyncio
    async def test_with_no_telegram_config_the_window_is_consumed(self):
        """`send_infra_alert` answers False for THREE things — unconfigured,
        rate-limited, refused. A deployment with no Telegram target has
        nothing to deliver, so the send is DONE: without this branch the
        watch sits in the retry path, logging "no telegram config; skipping"
        every 5 min — three times over the 15 min driven here — until the
        dead-target bound widens it, instead of once per 6 h interval."""
        feed, pager = _Feed(_fleet(1)), _Pager(delivered=False)
        # Pinned rather than inherited from the environment: a developer with
        # a real .env would otherwise be testing the OTHER branch.
        with patch.object(RS.settings, "infra_alert_telegram_token", ""), \
             patch.object(RS.settings, "infra_alert_telegram_chat_id", ""), \
             patch.object(RS.settings, "admin_alert_telegram_token", ""), \
             patch.object(RS.settings, "admin_alert_telegram_chat_id", ""), \
             patch.object(RS, "_bridge_get", feed), \
             patch.object(RS, "_send_telegram", pager):
            for i in range(int(RS._FLEET_ALERT_RETRY_S // TICK) * 3):
                await RS._fleet_watch_once(now=1_000.0 + TICK * i)
        assert len(pager.sent) == 1, pager.sent

    @pytest.mark.asyncio
    async def test_a_dead_target_stops_being_retried_at_fault_speed(self, configured):
        """"Configured" is not "reachable". A revoked token, or the bot
        removed from the chat, answers 401/403 for good — and
        `infra_alerts_configured()` cannot tell that from a working target, so
        it lands in the refused-send retry path and stays there: one POST
        every 5 min for as long as the fault stands, ~72 across a 6 h episode
        where the old timestamp gate made one.

        So the retry is bounded: after `_FLEET_ALERT_DEAD_AFTER` CONSECUTIVE
        refusals the spacing widens to `rollout_fleet_alert_interval_s`. The
        short spacing keeps its whole purpose — a 429 or a brief Telegram
        outage clears well inside six attempts (30 min).
        """
        n, retry = RS._FLEET_ALERT_DEAD_AFTER, RS._FLEET_ALERT_RETRY_S
        interval = float(RS.settings.rollout_fleet_alert_interval_s)
        t0, feed, pager = 1_000.0, _Feed(_fleet(1)), _Pager(delivered=False)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i in range(n):
                await RS._fleet_watch_once(now=t0 + retry * i)
            assert len(pager.sent) == n, pager.sent
            last = t0 + retry * (n - 1)
            await RS._fleet_watch_once(now=last + retry)          # was enough; isn't now
            assert len(pager.sent) == n, pager.sent
            await RS._fleet_watch_once(now=last + interval - 1)
            assert len(pager.sent) == n, pager.sent
            await RS._fleet_watch_once(now=last + interval)       # …and then it tries again
        assert len(pager.sent) == n + 1, pager.sent

    @pytest.mark.asyncio
    async def test_a_worsening_pages_at_once_from_inside_the_widened_backoff(self, configured):
        """The bound is on REPETITION, not on news. A fault that gets worse
        while the target looks dead still goes out on the next tick — and it
        restarts the count, because it is a different page rather than another
        attempt at the same one, so the short spacing applies to it again."""
        n, retry = RS._FLEET_ALERT_DEAD_AFTER, RS._FLEET_ALERT_RETRY_S
        t0, feed, pager = 1_000.0, _Feed(_fleet(1)), _Pager(delivered=False)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i in range(n):
                await RS._fleet_watch_once(now=t0 + retry * i)
            assert len(pager.sent) == n, pager.sent
            worse = t0 + retry * (n - 1) + TICK                   # deep in the wide window
            feed.fleet = _fleet(83)
            await RS._fleet_watch_once(now=worse)
            assert len(pager.sent) == n + 1, pager.sent
            assert "83 slot(s)" in pager.sent[-1][1]
            await RS._fleet_watch_once(now=worse + retry)         # short spacing again
        assert len(pager.sent) == n + 2, pager.sent

    @pytest.mark.asyncio
    async def test_a_confirmed_send_resets_the_refusal_count(self, configured):
        """The count means "consecutive refusals since a message last landed",
        and the confirmed send is what makes that true.

        Pinned on the state rather than through a later page, deliberately: a
        confirmed send also clears `send_fail_grade`, so the next attempt takes
        the worsened branch and would reset the count whatever this arm did.
        The invariant is what lets the spacing be read without asking how the
        last attempt ended — five refusals, a delivery, then one refusal must
        be one, not six.
        """
        n, retry = RS._FLEET_ALERT_DEAD_AFTER, RS._FLEET_ALERT_RETRY_S
        t0, feed, pager = 1_000.0, _Feed(_fleet(1)), _Pager(delivered=False)
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            for i in range(n - 1):
                await RS._fleet_watch_once(now=t0 + retry * i)
            assert RS._FLEET_STATE["send_fail_count"] == n - 1
            pager.delivered = True
            await RS._fleet_watch_once(now=t0 + retry * (n - 1))
        assert len(pager.sent) == n, pager.sent
        assert RS._FLEET_STATE["send_fail_count"] == 0

    @pytest.mark.asyncio
    async def test_a_dead_target_stops_retrying_the_recovery_notice_too(self, configured):
        """Same bound on the closing half, for the same reason — a dead target
        refuses the notice exactly as forever as it refuses the page, and this
        half retries from a branch that runs on every tick of a healthy
        fleet."""
        n, retry = RS._FLEET_ALERT_DEAD_AFTER, RS._FLEET_ALERT_RETRY_S
        interval = float(RS.settings.rollout_fleet_alert_interval_s)
        t0, feed, pager = 1_000.0, _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=t0)                    # fault, delivered
            pager.delivered = False
            feed.fleet = CONVERGED
            for i in range(n):
                await RS._fleet_watch_once(now=t0 + 100.0 + retry * i)
            assert pager.levels == ["warning"] + ["info"] * n, pager.sent
            last = t0 + 100.0 + retry * (n - 1)
            await RS._fleet_watch_once(now=last + retry)
            assert len(pager.sent) == n + 1, pager.sent
            await RS._fleet_watch_once(now=last + interval)
        assert len(pager.sent) == n + 2, pager.sent

    @pytest.mark.asyncio
    async def test_a_refused_recovery_notice_is_retried_until_it_lands(self, configured):
        """A recovery nobody received leaves the operator believing the fleet
        is still split — the exact asymmetry the rule above removes for the
        warning half. The first pass cleared the state, discarded the verdict
        and lost the notice for good."""
        feed, pager = _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)               # fault, delivered
            pager.delivered = False
            feed.fleet = CONVERGED
            await RS._fleet_watch_once(now=1_100.0)               # recovery refused
            await RS._fleet_watch_once(now=1_100.0 + TICK)        # spaced, not spammed
            assert pager.levels == ["warning", "info"], pager.sent
            pager.delivered = True
            await RS._fleet_watch_once(now=1_100.0 + RS._FLEET_ALERT_RETRY_S)
            await RS._fleet_watch_once(now=1_100.0 + RS._FLEET_ALERT_RETRY_S + TICK)
        assert pager.levels == ["warning", "info", "info"], pager.sent

    @pytest.mark.asyncio
    async def test_no_recovery_notice_goes_out_while_the_fleet_is_faulted(self, configured):
        """An owed close cannot be delivered late into a faulted fleet.

        Not because anything drops it — the marker is deliberately RETAINED
        across a new fault (the NOTE in `_fleet_watch_once`, just above
        `grade = _fleet_alert_grade(...)`, says why dropping it is a no-op at
        best and a loss at worst, and
        test_a_close_still_owed_survives_a_fault_page_nobody_received pins the
        retention). It cannot go out because the only send site is the
        no-warnings branch, whose body is rendered from that tick's own
        reading: a close can never arrive stale.

        The last tick is the one that matters, and it is placed PAST the owed
        close's retry spacing (refused at 1100, so retryable from 1400) while
        the fleet is faulted — a recovery send reachable from the faulted path
        would fire there and show up as a fourth level.
        """
        feed, pager = _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)
            pager.delivered = False
            feed.fleet = CONVERGED
            await RS._fleet_watch_once(now=1_100.0)               # recovery refused
            pager.delivered = True
            feed.fleet = _fleet(83)
            await RS._fleet_watch_once(now=1_200.0)               # faulted again
            # Past the owed close's retry spacing, still faulted: silent.
            await RS._fleet_watch_once(now=1_200.0 + RS._FLEET_ALERT_RETRY_S)
        assert pager.levels == ["warning", "info", "warning"], pager.sent


    @pytest.mark.asyncio
    async def test_a_close_still_owed_survives_a_fault_page_nobody_received(self, configured):
        """The reason nothing DROPS the owed close when the fleet faults again.

        A recovery notice can only be sent from the no-warnings branch, and its
        body is rendered from that tick's own reading — so it can never go out
        stale or while the fleet is faulted, and discarding it on a new fault
        buys nothing. It does cost something: here the SECOND fault's page is
        refused, so the only message the operator ever received is the first
        fault. Dropping the marker on the new fault would leave that page
        unclosed forever — a narrower instance of the very bug this pair of
        rules removes."""
        feed, pager = _Feed(_fleet(1)), _Pager()
        with patch.object(RS, "_bridge_get", feed), patch.object(RS, "_send_telegram", pager):
            await RS._fleet_watch_once(now=1_000.0)               # fault, delivered
            pager.delivered = False
            feed.fleet = CONVERGED
            await RS._fleet_watch_once(now=1_100.0)               # recovery refused
            feed.fleet = _fleet(83)
            await RS._fleet_watch_once(now=1_200.0)               # fault page refused
            feed.fleet = CONVERGED
            pager.delivered = True
            await RS._fleet_watch_once(now=1_100.0 + RS._FLEET_ALERT_RETRY_S)
        assert pager.levels == ["warning", "info", "warning", "info"], pager.sent


# ─── one runner per tick ──────────────────────────────────────────


class _StopAfterTicks(Exception):
    """Ends the reconciler loop from inside its tick sleep, deterministically
    and with no wall clock. Not CancelledError: that is the shutdown path and
    has its own test below."""


class TestLeaderGate:
    """Behavioural, not a source probe. A source probe that only asserts "an
    acquire exists above the work" survives hoisting the acquire ABOVE
    `while True` — one claim at boot, never renewed, the row expiring
    mid-run, and the replica that answered False once skipping forever.
    """

    ARMS = ("_reconcile_once", "_convergence_sweep_once", "_fleet_watch_once")

    @classmethod
    async def _drive(cls, ticks: int, acquire):
        """Run the real loop for exactly `ticks` ticks. Returns
        (acquire_mock, per-tick snapshots of each arm's await count, the
        values passed to sleep)."""
        arms = {name: AsyncMock() for name in cls.ARMS}
        slept: list[float] = []
        snapshots: list[dict] = []

        async def _sleep(seconds):
            slept.append(seconds)
            snapshots.append({k: m.await_count for k, m in arms.items()})
            if len(slept) >= ticks:
                raise _StopAfterTicks

        with patch("app.services.infra_lease.acquire_lease", acquire), \
             patch("asyncio.sleep", _sleep), \
             patch.multiple(RS, **arms):
            with pytest.raises(_StopAfterTicks):
                await RS.rollout_reconciler_loop()
        return snapshots, slept

    @pytest.mark.asyncio
    async def test_a_replica_without_the_lease_does_no_work_at_all(self):
        snaps, _ = await self._drive(1, AsyncMock(return_value=False))
        assert snaps == [{k: 0 for k in self.ARMS}], snaps

    @pytest.mark.asyncio
    async def test_the_lease_holder_runs_every_arm_of_the_tick(self):
        """Gating must not quietly drop one of the three responsibilities."""
        snaps, _ = await self._drive(1, AsyncMock(return_value=True))
        assert snaps == [{k: 1 for k in self.ARMS}], snaps

    @pytest.mark.asyncio
    async def test_the_lease_is_reacquired_every_tick_with_a_three_times_ttl(self):
        """Renewal, per tick, for real — three ticks driven through the loop.

        A gate hoisted out of `while True` passes a source probe and fails
        here: it would acquire ONCE. And the election must be re-run every
        tick in both directions — the replica that loses tick 1 must still
        take tick 2 when the holder dies, which is the whole failover path.
        """
        acq = AsyncMock(side_effect=[False, True, False])
        snaps, slept = await self._drive(3, acq)
        assert acq.await_count == 3, "one election per tick, not one per boot"
        for call in acq.await_args_list:
            assert call.args == ("rollout_reconciler",), call
            # 3x, so one missed renewal cannot hand the lease over mid-tick.
            assert call.kwargs["ttl_s"] == 3 * RS._RECONCILER_TICK_S, call
        assert snaps == [
            {k: 0 for k in self.ARMS},   # lost: nothing ran
            {k: 1 for k in self.ARMS},   # won: every arm ran
            {k: 1 for k in self.ARMS},   # lost again: nothing more ran
        ], snaps
        assert slept == [RS._RECONCILER_TICK_S] * 3, slept

    @pytest.mark.asyncio
    async def test_a_graceful_shutdown_hands_the_lease_back(self):
        """Every one of the three arms is behind the gate now, so without a
        release a deploy idles rollout resume, stuck-rollout orphaning AND the
        fleet watch for up to ttl = 3x the tick (90 s) — where before the gate
        both replicas reconciled from boot. Same measurement as
        container_reconciler_loop's release (2026-09-16 post-deploy gaps)."""
        rel = AsyncMock(return_value=True)
        arms = {name: AsyncMock() for name in self.ARMS}
        with patch("app.services.infra_lease.acquire_lease", AsyncMock(return_value=True)), \
             patch("app.services.infra_lease.release_lease", rel), \
             patch.multiple(RS, **arms):
            task = asyncio.ensure_future(RS.rollout_reconciler_loop())
            await asyncio.sleep(0.2)          # into the tick sleep
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        rel.assert_awaited_once()
        assert rel.await_args[0][0] == "rollout_reconciler"
