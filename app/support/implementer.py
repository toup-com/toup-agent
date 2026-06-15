"""Implement & verify a fix — runs ONLY after admin approval.

Hard guardrails (all enforced here AND by the pipeline):
  * Runs only when ``settings.support_implement_enabled`` is True. Off by
    default so the diagnosis+approval loop can ship before the autonomous
    code-writer is trusted, and so it is inert in environments (e.g. the
    Railway platform-api) that have no git checkout / gh credentials.
  * Never writes to the base branch (main) or prod. It branches off
    ``settings.support_pr_base_branch`` and opens ONE PR per issue.
  * Edits are constrained to the diagnosis's ``affected_files`` and applied
    by EXACT string replacement (no fuzzy patching) — an edit that doesn't
    match uniquely is skipped and reported, never force-applied.
  * The fix must respect the skill invariants (passed into the prompt as
    "MUST NOT violate"); they are also restated in the PR body.
  * Rollback path: the PR is reviewable and revertable; the working branch
    is left intact on failure for inspection (delete branch to roll back).

The caller (pipeline) persists status/events; this module is pure execution
and returns a structured result.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import settings
from app.support import skills_index
from app.support.llm import ask_json

logger = logging.getLogger("support.implementer")

_SYSTEM = (
    "You are implementing an APPROVED bug fix for the Toup platform. You are "
    "given the diagnosis (root cause, proposed fix, affected files), the "
    "invariants the fix MUST NOT violate, and real excerpts of the files. "
    "Produce minimal, surgical edits as exact string replacements.\n\n"
    "RULES:\n"
    "- Only edit files listed in affected_files.\n"
    "- Each edit's \"find\" must be a UNIQUE, verbatim substring of the current "
    "file (copy enough surrounding context to be unique). \"replace\" is the new "
    "text. Do not reformat unrelated code.\n"
    "- Never violate a listed invariant/footgun.\n"
    "- If you cannot produce a safe minimal edit, return an empty edits list and "
    "explain in commit_message why.\n\n"
    "Output JSON: {\"edits\": [{\"file\": \"repo/rel/path\", \"find\": \"...\", "
    "\"replace\": \"...\", \"why\": \"...\"}], \"commit_message\": \"conventional-commit "
    "one-liner\", \"pr_title\": \"...\", \"notes\": \"...\"}."
)


@dataclass
class ImplementResult:
    ok: bool
    reason: str = ""
    branch: Optional[str] = None
    pr_url: Optional[str] = None
    edits_applied: list = field(default_factory=list)
    edits_failed: list = field(default_factory=list)
    verification: dict = field(default_factory=dict)
    error: Optional[str] = None


def _slug(text: str, n: int = 32) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (s[:n] or "fix").rstrip("-")


def _run(cmd: list, cwd: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def _preconditions(repo_dir: str) -> Optional[str]:
    if not getattr(settings, "support_implement_enabled", False):
        return ("support_implement_enabled is False — implementation is gated off "
                "(diagnosis+approval only). Enable it in an environment with a git "
                "checkout + gh credentials to let the agent open fix PRs.")
    if not Path(repo_dir, ".git").exists():
        return f"{repo_dir} is not a git repository — no checkout to implement against."
    return None


def _git_excerpt(repo_dir: str, rel: str, max_lines: int = 400) -> Optional[str]:
    p = Path(repo_dir) / rel
    if not p.is_file():
        return None
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[:max_lines])


async def implement(
    *,
    user_id: str,
    issue_id: str,
    symptom: str,
    diagnosis: dict,
) -> ImplementResult:
    repo_dir = str(getattr(settings, "support_repo_dir", None) or skills_index.repo_root())
    base = getattr(settings, "support_pr_base_branch", "main")

    pre = _preconditions(repo_dir)
    if pre:
        return ImplementResult(ok=False, reason=pre)

    affected = [f for f in (diagnosis.get("affected_files") or []) if isinstance(f, str)]
    if not affected:
        return ImplementResult(ok=False, reason="Diagnosis has no affected_files to edit.")

    # 1) Ask the LLM for surgical edits, grounded in the real current files.
    excerpts = []
    for rel in affected[:10]:
        ex = _git_excerpt(repo_dir, rel)
        if ex is not None:
            excerpts.append(f"===== {rel} =====\n{ex}")
    invariants = diagnosis.get("invariants_to_preserve") or []
    prompt = (
        f"SYMPTOM: {symptom}\n\n"
        f"DIAGNOSIS:\n{diagnosis.get('root_cause','')}\n\n"
        f"PROPOSED FIX:\n{diagnosis.get('proposed_fix','')}\n\n"
        f"AFFECTED FILES: {affected}\n\n"
        f"INVARIANTS THE FIX MUST NOT VIOLATE:\n- " + "\n- ".join(map(str, invariants)) + "\n\n"
        f"CURRENT FILE CONTENTS:\n" + "\n\n".join(excerpts)
    )
    plan = await ask_json(
        user_id=user_id,
        operation_type="system.support.implement",
        system=_SYSTEM,
        prompt=prompt,
        max_tokens=2600,
        timeout=120,
    )
    if not plan or not plan.get("edits"):
        return ImplementResult(
            ok=False,
            reason="LLM produced no applicable edits.",
            error=(plan or {}).get("notes") if plan else "LLM unavailable",
        )

    # 2..N) Branch, apply, verify, PR — all blocking git work in a thread.
    import asyncio
    return await asyncio.to_thread(
        _apply_verify_pr, repo_dir, base, issue_id, symptom, diagnosis, plan,
    )


def _apply_verify_pr(repo_dir, base, issue_id, symptom, diagnosis, plan) -> ImplementResult:
    branch = f"support/issue-{issue_id[:8]}-{_slug(plan.get('pr_title') or symptom)}"
    affected = set(diagnosis.get("affected_files") or [])
    applied: list = []
    failed: list = []
    try:
        # Fresh branch off the up-to-date base. Never commit onto base itself.
        _run(["git", "fetch", "origin", base], repo_dir)
        co = _run(["git", "checkout", "-B", branch, f"origin/{base}"], repo_dir)
        if co.returncode != 0:
            # Fall back to local base if origin/base isn't fetchable.
            co = _run(["git", "checkout", "-B", branch, base], repo_dir)
            if co.returncode != 0:
                return ImplementResult(ok=False, reason="could not create branch",
                                       branch=branch, error=co.stderr.strip())

        # Apply each edit by exact, unique string replacement.
        for ed in plan["edits"]:
            rel, find, replace = ed.get("file"), ed.get("find"), ed.get("replace")
            if rel not in affected:
                failed.append({**ed, "_reason": "file not in affected_files"})
                continue
            p = Path(repo_dir) / rel
            if not p.is_file():
                failed.append({**ed, "_reason": "file missing"})
                continue
            content = p.read_text(encoding="utf-8")
            if not find or content.count(find) != 1:
                failed.append({**ed, "_reason": f"find not unique (count={content.count(find) if find else 0})"})
                continue
            p.write_text(content.replace(find, replace, 1), encoding="utf-8")
            applied.append({"file": rel, "why": ed.get("why", "")})

        if not applied:
            _run(["git", "checkout", base], repo_dir)
            return ImplementResult(ok=False, reason="no edits applied (all skipped)",
                                   branch=branch, edits_failed=failed)

        # Verify.
        verification = _verify(repo_dir)

        # Commit.
        _run(["git", "add", *(e["file"] for e in applied)], repo_dir)
        commit_msg = plan.get("commit_message") or f"fix: {symptom[:60]}"
        body = _commit_trailer(issue_id, diagnosis)
        cm = _run(["git", "commit", "-m", commit_msg, "-m", body], repo_dir)
        if cm.returncode != 0:
            return ImplementResult(ok=False, reason="commit failed", branch=branch,
                                   edits_applied=applied, edits_failed=failed,
                                   verification=verification, error=cm.stderr.strip())

        # Open PR (push + gh). Only if verification passed and gh is allowed.
        pr_url = None
        if verification.get("passed", False) and getattr(settings, "support_open_pr", True):
            pr_url = _open_pr(repo_dir, branch, base, plan, diagnosis, issue_id)

        return ImplementResult(
            ok=verification.get("passed", False),
            reason="implemented" if verification.get("passed") else "verification failed",
            branch=branch, pr_url=pr_url,
            edits_applied=applied, edits_failed=failed, verification=verification,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("[support.implementer] failed")
        return ImplementResult(ok=False, reason="exception", branch=branch,
                               edits_applied=applied, edits_failed=failed, error=str(e))


def _verify(repo_dir: str) -> dict:
    """Run the configured verification commands; capture pass/fail + output tail."""
    cmds = getattr(settings, "support_verify_commands", None) or [
        # Safe default: prove the platform entrypoint still imports.
        "python -c \"import platform_main\"",
    ]
    results = []
    passed = True
    for c in cmds:
        try:
            r = _run(["bash", "-lc", c], repo_dir, timeout=900)
            ok = r.returncode == 0
            passed = passed and ok
            results.append({
                "command": c, "ok": ok, "returncode": r.returncode,
                "output_tail": (r.stdout + r.stderr)[-1500:],
            })
        except Exception as e:
            passed = False
            results.append({"command": c, "ok": False, "error": str(e)})
    return {"passed": passed, "results": results}


def _commit_trailer(issue_id: str, diagnosis: dict) -> str:
    skills = ", ".join(diagnosis.get("skills_referenced") or [])
    return (
        f"Support issue: {issue_id}\n"
        f"Skill(s) relied on: {skills}\n"
        f"Rollback: {diagnosis.get('rollback_path','revert this PR')}\n\n"
        "Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
    )


def _open_pr(repo_dir, branch, base, plan, diagnosis, issue_id) -> Optional[str]:
    push = _run(["git", "push", "-u", "origin", branch], repo_dir)
    if push.returncode != 0:
        logger.warning("[support.implementer] push failed: %s", push.stderr.strip())
        return None
    # gh must be installed + authed in this environment.
    title = plan.get("pr_title") or "fix: support-agent automated fix"
    skills = ", ".join(diagnosis.get("skills_referenced") or [])
    pr_body = (
        f"Automated fix opened by the Toup maintenance/support agent for issue `{issue_id}`.\n\n"
        f"## Diagnosis\n"
        f"**Root cause:** {diagnosis.get('root_cause','')}\n\n"
        f"**Blast radius:** {diagnosis.get('blast_radius','')}\n\n"
        f"**Proposed fix:** {diagnosis.get('proposed_fix','')}\n\n"
        f"**Affected files:** {', '.join(diagnosis.get('affected_files') or [])}\n\n"
        f"## Skill(s) relied on (source of truth)\n{skills}\n\n"
        f"## Invariants preserved\n- "
        + "\n- ".join(map(str, diagnosis.get('invariants_to_preserve') or ['(none recorded)']))
        + f"\n\n## Test plan\n{diagnosis.get('test_plan','')}\n\n"
        f"## Rollback path\n{diagnosis.get('rollback_path','Revert this PR; no migration/data changes unless noted.')}\n\n"
        f"---\n🤖 Generated with [Claude Code](https://claude.com/claude-code)"
    )
    pr = _run(["gh", "pr", "create", "--base", base, "--head", branch,
               "--title", title, "--body", pr_body], repo_dir)
    if pr.returncode != 0:
        logger.warning("[support.implementer] gh pr create failed: %s", pr.stderr.strip())
        return None
    return (pr.stdout or "").strip() or None
