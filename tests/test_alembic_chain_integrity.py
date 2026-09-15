"""Regression guard: the alembic chain is a single unbranched line.

Why this exists, in one incident. `feat/unlimited-entitlement` was cut off
`main` when head was 101 and added revisions "102" and "103". While it was in
flight, the 2026-09-12 incident's wave-2 branch merged its own "102"
(infra_leases) and "103" (agent_probe_state) off the same 101. Both branches
were internally consistent; the MERGE was the defect, and nothing in the repo
could see it:

  * `tsc`/pytest never load the versions directory as a chain;
  * alembic itself does not hard-fail on a duplicate id — it emits a
    `UserWarning: Revision 102 is present more than once`, resolves
    `get_revision("103")` to whichever module loaded last, and only raises
    `RevisionError: Requested revision 103 overlaps with other requested
    revisions 102` at `upgrade head` time;
  * and `Dockerfile`'s boot line is
    `(cd backend && alembic upgrade head) || echo '[boot] alembic upgrade
    failed — continuing without applying migrations'`, so that raise prints
    one line and the app serves traffic against a schema its ORM models
    disagree with.

For a billing branch the downstream shape is: `select(AppleSubscription)`
emits `grandfathered_at`, Postgres answers `UndefinedColumn`, and every App
Store Server Notification raises into `iap.py`'s swallow-and-200 — i.e. a
renewal lost forever, with no retry and no alert.

So the invariant is checked here, cheaply, on every run: one head, no
duplicate ids, every `down_revision` resolvable. A branch that collides with
`main` fails this the moment the two are in the same tree.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "alembic" / "versions"


def _literal(node: ast.AST):
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _revisions() -> dict[str, dict]:
    """{path: {"revision": str|None, "down_revision": str|tuple|None}}."""
    out: dict[str, dict] = {}
    for path in sorted(_MIGRATIONS_DIR.glob("*.py")):
        if path.name.startswith("__"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: dict = {"revision": None, "down_revision": None}
        for node in tree.body:  # module level only — these are module globals
            # Both spellings are in this tree: bare `revision = "104"` and the
            # annotated `revision: str = '001_initial'` alembic's own template
            # emits. Reading only ast.Assign silently skips the annotated ones,
            # which makes every check below vacuous for those files.
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets = [node.target]
            else:
                continue
            for target in targets:
                if isinstance(target, ast.Name) and target.id in found:
                    found[target.id] = _literal(node.value)
        out[path.name] = found
    return out


def test_every_migration_declares_a_revision():
    missing = [name for name, r in _revisions().items() if not r["revision"]]
    assert not missing, f"migration(s) with no module-level `revision`: {missing}"


def test_no_duplicate_revision_ids():
    """Two files claiming the same id is the merge collision described above.

    alembic only WARNS on this; the failure surfaces later, at `upgrade head`,
    inside a boot line that swallows it.
    """
    by_id: dict[str, list[str]] = {}
    for name, r in _revisions().items():
        by_id.setdefault(str(r["revision"]), []).append(name)
    dupes = {rev: files for rev, files in by_id.items() if len(files) > 1}
    assert not dupes, (
        "duplicate alembic revision id(s) — rebase onto main and renumber:\n"
        + "\n".join(f"  {rev}: {files}" for rev, files in sorted(dupes.items()))
    )


def test_every_down_revision_resolves():
    revs = _revisions()
    known = {str(r["revision"]) for r in revs.values()}
    orphans = []
    for name, r in revs.items():
        down = r["down_revision"]
        if down is None:
            continue
        parents = down if isinstance(down, (tuple, list)) else [down]
        for parent in parents:
            if str(parent) not in known:
                orphans.append((name, parent))
    assert not orphans, f"down_revision pointing at nothing: {orphans}"


def test_the_chain_has_exactly_one_head():
    """A second head means `upgrade head` is ambiguous and alembic refuses.

    This is the arm that catches a collision even when the two branches
    happened to pick different ids.
    """
    revs = _revisions()
    referenced: Counter[str] = Counter()
    for r in revs.values():
        down = r["down_revision"]
        if down is None:
            continue
        parents = down if isinstance(down, (tuple, list)) else [down]
        for parent in parents:
            referenced[str(parent)] += 1

    heads = sorted(
        str(r["revision"]) for r in revs.values() if referenced[str(r["revision"])] == 0
    )
    assert len(heads) == 1, f"expected exactly one alembic head, found {heads}"


def test_no_revision_is_the_parent_of_two_migrations():
    """A fork is how two ids that do NOT collide still produce two heads.

    Named separately from the head test so the failure says WHERE the tree
    split, not just that it did.
    """
    revs = _revisions()
    children: dict[str, list[str]] = {}
    for name, r in revs.items():
        down = r["down_revision"]
        if down is None:
            continue
        parents = down if isinstance(down, (tuple, list)) else [down]
        for parent in parents:
            children.setdefault(str(parent), []).append(name)
    forks = {p: c for p, c in children.items() if len(c) > 1}
    assert not forks, (
        "alembic chain forks — two migrations share a parent:\n"
        + "\n".join(f"  {p} -> {c}" for p, c in sorted(forks.items()))
    )
