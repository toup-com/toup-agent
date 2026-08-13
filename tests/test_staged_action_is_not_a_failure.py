"""A staged action is waiting, not failed.

Founder session, 2026-08-12. "Create a spreadsheet called Toup Demo
Data..." produced a job card reading **Failed** — while the spreadsheet
itself was created successfully and is sitting in the user's Drive.

The sequence:

  create_job  -> "Create demo expense spreadsheet", 3 steps
  create_spreadsheet -> `confirmation_required` (elevation: true, so the
                        dispatcher STAGES it and returns a card instead
                        of running it)
  update_job  -> {"ok": true, "status": "failed", "completed_steps": 0}

From the model's seat that looked right: the step had not completed. But
`confirmation_required` means *not yet*, not *no*. The user then pressed
Send, the tool ran, the sheet was created — and nothing went back to the
job, because `connector_pending_actions` has no `job_id` column and the
approval route has no way to reach the job that wrapped it.

So the card is permanently wrong in the worst direction: it says failed
about work that succeeded.

Two rules, both in the prompt because that is where the defect is:

  1. A single connector call does not get a job. The approval card IS
     the status; a job around it is a second status with no way to stay
     in sync with the first.
  2. `confirmation_required` is never reported as a failure.

The structural fix — a `job_id` on the pending action, resolved when the
approval executes — is the real one and is not in this change. These
rules stop the model MANUFACTURING the inconsistency in the meantime.
"""

from __future__ import annotations

import pathlib
import re

AR = pathlib.Path("app/agent/agent_runner.py")


def _src() -> str:
    return AR.read_text()


def test_confirmation_required_is_explicitly_not_a_failure():
    src = _src().lower()
    assert "confirmation_required" in src, (
        "the prompt never mentions confirmation_required, so the model has "
        "only the tool result to interpret — and it read it as a failure"
    )
    # The rule has to forbid the specific action, not merely describe the
    # state: "it is staged" alone still leaves update_job(failed) open.
    assert "never `update_job` it to `failed`" in _src() or "not failed" in src, (
        "the rule must forbid marking the job failed, not just explain "
        "what confirmation_required means"
    )


def test_create_job_rule_excludes_a_single_connector_call():
    src = _src()
    i = src.find("call `create_job` FIRST")
    assert i != -1, "the create_job rule moved — update this guard"
    window = src[i:i + 700]
    assert "NOT for a single connector call" in window, (
        "the create_job rule still tells the model to wrap one connector "
        "call in a job, which is how a succeeded write got a Failed card"
    )


def test_the_two_rules_sit_together_in_the_job_section():
    """Both rules qualify the same instruction; separating them lets one
    be edited away without the other becoming obviously wrong."""
    src = _src()
    a = src.find("NOT for a single connector call")
    b = src.find("confirmation_required")
    assert a != -1 and b != -1
    assert abs(a - b) < 600, (
        "the staged-action rule drifted away from the create_job rule it "
        "qualifies"
    )


def test_pending_actions_still_have_no_job_link():
    """Pins the reason these are PROMPT rules rather than a code fix.

    When someone adds `job_id` to connector_pending_actions and resolves
    the job on approval, this test fails — and that is the signal to
    replace the prompt rules with the structural guarantee.
    """
    model = pathlib.Path("app/db/models/connectors.py").read_text()
    block = model[model.index('__tablename__ = "connector_pending_actions"'):]
    block = block[:block.index("class ", 10)] if "class " in block[10:] else block
    assert not re.search(r"^\s*job_id", block, re.M), (
        "connector_pending_actions now carries job_id — the approval path "
        "can reach its job, so replace these prompt rules with the real fix: "
        "resolve the job when the staged action executes"
    )
