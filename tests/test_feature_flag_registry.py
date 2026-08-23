"""
Feature-flag registry — the properties a percentage ramp has to have.

Pure-function, same reason as test_onboarding_v2_feature_flag.py: no DB, so it
runs everywhere. The DB override path is one `select` and a `commit`; what is
worth pinning is the bucketing, because every one of these properties is
invisible until a ramp is already in flight and each failure mode looks like a
different bug entirely:

  * not uniform     -> "10%" ships to 3% or 30% and the ramp means nothing
  * not monotone    -> going 10 -> 50 REMOVES users who had the feature, so a
                       ramp UP reads to them as a regression that undid itself
  * correlated      -> the second flag's 10% is the SAME people as the first's,
                       so two experiments silently run on one cohort
  * anonymous on    -> a logged-out visitor gets a partial rollout and flips
                       between cohorts on every refresh
"""

from __future__ import annotations

import pytest

from app.services.feature_flags import (
    FLAGS, FlagSpec, _clamp_pct, is_enabled_for, is_onboarding_v2_enabled_for,
)

USERS = [f"user-{i}" for i in range(10_000)]


def cohort(pct: int, salt: str) -> set[str]:
    return {u for u in USERS if is_enabled_for(u, pct, salt)}


# ── The registry itself ──────────────────────────────────────────────

def test_registry_holds_both_shipped_flags():
    # Round 26 added `automations` (dark launch, env floor 0). Every
    # entry here is a published wire name — see the comment below.
    assert set(FLAGS) == {"onboarding_v2", "web_mobile_shell", "automations"}
    for name, spec in FLAGS.items():
        assert isinstance(spec, FlagSpec)
        # The wire name is what the frontend reads out of
        # GET /api/system/feature-flags. A mismatch here turns the flag off for
        # every deployed client while the admin toggle reports it on.
        assert spec.name == name
        assert spec.setting_key and spec.env_attr


def test_onboarding_v2_bucketing_is_unchanged_by_the_registry():
    # Its salt is "" precisely so generalising the module did not reshuffle a
    # rollout that is already in flight. If this fails, live users moved
    # between two onboarding flows.
    assert FLAGS["onboarding_v2"].salt == ""
    for seed in ("user-1", "user-500", "203.0.113.9"):
        assert is_onboarding_v2_enabled_for(seed, 37) == is_enabled_for(seed, 37, "")


def test_env_attr_exists_on_settings():
    from app.config import settings
    for spec in FLAGS.values():
        # A typo'd attr silently reads 0 via getattr's default, i.e. the flag
        # is permanently off and the env var that is supposed to set the
        # fresh-deploy floor does nothing.
        assert hasattr(settings, spec.env_attr), spec.env_attr


# ── Bucketing properties ─────────────────────────────────────────────

@pytest.mark.parametrize("pct", [10, 25, 50, 75])
def test_bucket_share_is_within_two_points_of_the_requested_pct(pct):
    share = len(cohort(pct, "web_mobile_shell")) / len(USERS) * 100
    assert abs(share - pct) < 2, f"asked for {pct}%, bucketed {share:.2f}%"


def test_ramping_up_never_removes_anyone():
    a = cohort(10, "web_mobile_shell")
    b = cohort(50, "web_mobile_shell")
    c = cohort(100, "web_mobile_shell")
    assert a < b < c


def test_two_flags_do_not_pick_the_same_people():
    mobile = cohort(10, FLAGS["web_mobile_shell"].salt)
    onboarding = cohort(10, FLAGS["onboarding_v2"].salt)
    overlap = len(mobile & onboarding) / len(mobile)
    # Independent 10% cohorts overlap ~10%. Sharing a seed would make this 1.0.
    assert overlap < 0.2, f"cohorts overlap {overlap:.0%} — the salt is not working"


def test_zero_is_off_for_everyone_and_hundred_is_on_for_everyone():
    assert cohort(0, "web_mobile_shell") == set()
    assert len(cohort(100, "web_mobile_shell")) == len(USERS)


def test_anonymous_callers_are_off_below_full_launch():
    for seed in (None, ""):
        assert is_enabled_for(seed, 0, "x") is False
        assert is_enabled_for(seed, 50, "x") is False
        # At 100 there is no cohort left to be outside of.
        assert is_enabled_for(seed, 100, "x") is True


def test_out_of_range_percentages_clamp():
    assert _clamp_pct(-5) == 0
    assert _clamp_pct(140) == 100
