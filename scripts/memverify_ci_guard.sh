#!/usr/bin/env bash
#
# Decide whether the agent memory verification suite is REQUIRED to run.
#
# Why this is a committed script and not four lines of YAML
# ---------------------------------------------------------
# Until 2026-08-06 the CI step began:
#
#     if [ -z "$OPENAI_API_KEY" ]; then
#       echo "::warning title=memory verification skipped::..."
#       exit 0
#     fi
#
# The repo has never had an OPENAI_API_KEY secret, so that branch was the ONLY
# branch ever taken: the suite has never run, and every build reported a green
# check for it. An annotation that exits 0 is indistinguishable from a pass in
# the checks UI, which is exactly how it stayed invisible.
#
# Living in a file means the decision is executable — and therefore testable —
# off a GitHub runner. tests/test_memverify_metrics.py runs THIS script under
# every combination of (secret present/absent) x (fork/same-repo/push) and
# asserts the exit code and the emitted decision.
#
# Contract
# --------
#   stdin/args : none
#   env        : OPENAI_API_KEY        - the secret, forwarded by the workflow
#                MEMVERIFY_IS_FORK_PR  - "true" only for a pull_request whose
#                                        head repo differs from this one
#                GITHUB_OUTPUT         - optional; the Actions step-output file
#   stdout     : a human-readable line plus, when GITHUB_OUTPUT is set,
#                `decision=<run|skip>` appended to that file
#   exit 0     : decision=run   -> the caller MUST run the suite
#   exit 0     : decision=skip  -> tolerated; a fork PR cannot have the secret
#   exit 1     : the secret is missing somewhere it is required
#
# The key value itself is never echoed, only its presence.

set -euo pipefail

key="${OPENAI_API_KEY:-}"
is_fork_pr="${MEMVERIFY_IS_FORK_PR:-}"

emit() {
  # $1 = decision. Written to the step-output file when running under Actions.
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    printf 'decision=%s\n' "$1" >>"$GITHUB_OUTPUT"
  fi
  printf 'decision=%s\n' "$1"
}

if [ -n "$key" ]; then
  emit run
  exit 0
fi

# GitHub deliberately withholds secrets from workflows triggered by a pull
# request opened from a fork, so a contributor there CANNOT satisfy this and
# must not be blocked by it. This is the one tolerated skip, and because the
# suite then becomes a separate `if:`-gated step, the checks UI shows it grey
# ("skipped") rather than green.
if [ "$is_fork_pr" = "true" ]; then
  echo "::notice title=memory verification skipped (fork PR)::OPENAI_API_KEY is not available to fork pull requests, so the agent memory suite did not run. A maintainer must re-run it on a same-repo branch before merging."
  emit skip
  exit 0
fi

# Everything else — push to main, same-repo PR, workflow_dispatch — is a run
# where the secret is available in principle. Failing here is the entire point
# of this file: a missing secret is a broken CI configuration, not a pass.
echo "::error title=memory verification did not run::OPENAI_API_KEY is not set for this repository, so the agent memory verification suite (backend/tests/memverify, 67 labeled scenarios) could not run. This is the ONLY place the agent memory system executes end to end — pgvector cosine, tsvector keyword search, the entity graph, the real extractor and real embeddings — and it has never run in CI. Fix: add the OPENAI_API_KEY repository secret (Settings -> Secrets and variables -> Actions). Fork pull requests are exempt automatically."
emit fail
exit 1
