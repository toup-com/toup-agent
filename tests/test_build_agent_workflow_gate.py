"""A manual build of a branch must not roll that branch to real users.

WHY THIS EXISTS — 2026-09-06 13:20–14:29:

    13:20  Build Agent Image, workflow_dispatch, branch codex/mobile-voice-agent-runtime
           -> ghcr.io/toup-com/toup-agent:a962b7340717
    13:31  POST /api/admin/rollout/start with that tag
    13:39  7 dedicated tenants on a branch image
    13:41  the bridge blue-greens 41 real users' pool slots onto it
    14:29  paused at 41/72; those users stayed on it for nine hours

`skip_rollout` defaulted to 'false', so the WORKFLOW's own default for a manual
branch build was: notify Railway and roll the fleet. Nothing about the ref was
checked. The safe direction for a manual build is the opposite one — build the
image, name it, and let a human start the rollout with the tag in hand — so the
default is inverted here and a dispatch rollout additionally requires the
default branch.

This test reads the workflow file; there is no other executable check on it in
this repo (no actionlint in CI).

Run:
  cd backend && PYTHONPATH=. pytest tests/test_build_agent_workflow_gate.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

WF_PATH = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "build-agent.yml"


@pytest.fixture(scope="module")
def wf():
    assert WF_PATH.exists(), f"missing {WF_PATH}"
    # `on:` is parsed by PyYAML 1.1 rules as the boolean True — keep both.
    return yaml.safe_load(WF_PATH.read_text())


@pytest.fixture(scope="module")
def text():
    return WF_PATH.read_text()


def _triggers(wf):
    return wf.get("on") or wf.get(True)


def _rollout_steps(wf):
    steps = wf["jobs"]["build-and-rollout"]["steps"]
    return [s for s in steps if "rollout" in (s.get("name") or "").lower()
            and "if" in s]


class TestDispatchDefaults:
    def test_a_manual_build_does_not_roll_out_by_default(self):
        wf = yaml.safe_load(WF_PATH.read_text())
        inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
        assert str(inputs["skip_rollout"]["default"]).lower() == "true", (
            "workflow_dispatch must build without rolling unless the operator "
            "asks — a962b7340717 reached 41 users through this default"
        )

    def test_a_named_ref_can_be_built(self):
        wf = yaml.safe_load(WF_PATH.read_text())
        inputs = _triggers(wf)["workflow_dispatch"]["inputs"]
        assert "ref" in inputs, "operators need to name the ref they are building"

    def test_the_checkout_honours_the_named_ref(self, text):
        assert "actions/checkout@v4" in text
        assert "inputs.ref" in text, "the ref input must reach actions/checkout"

    def test_a_blank_ref_checks_out_the_run_sha_not_the_branch_tip(self, text):
        """`github.ref` re-resolves at checkout time, so two pushes a minute
        apart could build a tree the image tag does not name. `github.sha` is
        what checkout does with no ref at all."""
        assert "github.event.inputs.ref || github.sha" in text

    def test_the_tag_is_computed_from_the_checked_out_tree(self, text):
        """With a `ref` input, $GITHUB_SHA is the ref the RUN started on, not
        the one being built. An image tagged with a commit it does not contain
        is read as truth by every rollout and every incident timeline."""
        block = text.split("Compute tags", 1)[1].split("- name:", 1)[0]
        assert "git rev-parse HEAD" in block
        assert "${GITHUB_SHA::12}" not in block


class TestRolloutGating:
    def test_every_rollout_step_is_gated(self):
        wf = yaml.safe_load(WF_PATH.read_text())
        steps = _rollout_steps(wf)
        assert len(steps) >= 2, "expected the notify step and the completion gate"
        for s in steps:
            cond = s["if"]
            assert "skip rollout" in cond, f"{s['name']}: [skip rollout] not honoured"
            assert "skip_rollout" in cond, f"{s['name']}: dispatch input not honoured"

    def test_a_dispatch_rollout_requires_the_default_branch(self):
        """The only automatic path to the fleet is a merge to main. A manual
        rollout of a branch image is still possible — by hand, with the tag —
        but the workflow will not do it for you."""
        wf = yaml.safe_load(WF_PATH.read_text())
        for s in _rollout_steps(wf):
            cond = s["if"]
            assert "default_branch" in cond or "refs/heads/main" in cond, (
                f"{s['name']}: a workflow_dispatch rollout must be pinned to "
                "the default branch"
            )


class TestDocumentedSemantics:
    def test_the_header_says_skip_rollout_still_publishes_an_image(self, text):
        head = text.split("jobs:")[0].lower()
        assert "[skip rollout]" in head, "the marker's semantics belong in the header"
        assert "ghcr" in head, (
            "the header must say a skipped rollout STILL pushes an uncanaried "
            "image to GHCR — e698ea78 sat there and nobody knew"
        )
        assert "uncanaried" in head or "not canaried" in head
