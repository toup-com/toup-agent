"""Round 12 — the Expo reclamation script.

The script deletes hundreds of megabytes as root inside a tenant's container,
so the properties that matter are about what it must NOT do:

  * dry-run by default — measuring must never delete;
  * only regenerable machinery — never source, never `storage/`, never the
    SQLite file, never `.git`, never a lockfile;
  * never follow a symlink out of the apps root;
  * never touch the HTML app root;
  * skip an app whose dev server is live.

`test_apply_actually_reclaims` is the control: it proves the script CAN
delete, so the "did not delete" assertions above are not vacuous.
"""

from __future__ import annotations

import json
import os
import sys
import pathlib

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import cleanup_expo_apps as cleanup  # noqa: E402


def build_fake_app(root: pathlib.Path, name: str, *, nm_files: int = 40) -> pathlib.Path:
    """A miniature of the real shape: a lot of node_modules, a little source."""
    app = root / name
    (app / "node_modules" / "react" / "cjs").mkdir(parents=True)
    for i in range(nm_files):
        (app / "node_modules" / "react" / "cjs" / f"chunk-{i}.js").write_text("x" * 2048)
    (app / ".expo").mkdir()
    (app / ".expo" / "cache.json").write_text("y" * 4096)
    (app / "screens").mkdir()
    (app / "screens" / "HomeScreen.tsx").write_text("export default function Home(){}")
    (app / "storage").mkdir()
    (app / "storage" / "user.json").write_text('{"kept": true}')
    (app / "app.db").write_text("SQLITE-ish")
    (app / "package.json").write_text('{"name":"x"}')
    (app / "package-lock.json").write_text('{"lockfileVersion":3}')
    (app / ".git").mkdir()
    (app / ".git" / "HEAD").write_text("ref: refs/heads/main")
    return app


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    root.mkdir()
    build_fake_app(root, "budget-tracker")
    build_fake_app(root, "workout-log", nm_files=20)
    # No live dev servers, and no stray metro caches from the host machine.
    monkeypatch.setattr(cleanup, "_listening_ports", lambda: set())
    monkeypatch.setattr(cleanup, "metro_tmp_caches", lambda: [])
    return root


def run(root, *argv):
    return cleanup.main(["--apps-root", str(root), *argv])


def survivors(app: pathlib.Path) -> set:
    return {p.name for p in app.iterdir()}


# ── 1. Dry run measures and does not delete ──────────────────────────

def test_default_is_a_dry_run(tree, capsys):
    assert run(tree) == 0
    out = capsys.readouterr().out
    assert "Dry run" in out
    assert "Reclaimable" in out
    assert (tree / "budget-tracker" / "node_modules").is_dir()
    assert (tree / "budget-tracker" / ".expo").is_dir()


def test_json_output_carries_the_numbers(tree, capsys):
    run(tree, "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is False
    assert payload["removed_bytes"] == 0
    assert payload["reclaimable_bytes"] > 0
    names = {a["app"] for a in payload["apps"]}
    assert names == {"budget-tracker", "workout-log"}
    for a in payload["apps"]:
        assert a["reclaimable_bytes"] > 0
        assert a["reclaimable_bytes"] <= a["total_bytes"]
        assert any(t.endswith("node_modules") for t in a["targets"])


# ── 2. Apply reclaims, and only the regenerable half ─────────────────

def test_apply_actually_reclaims(tree, capsys):
    """CONTROL for every 'was not deleted' assertion in this file."""
    run(tree, "--json")
    before = json.loads(capsys.readouterr().out)["reclaimable_bytes"]
    assert before > 0

    run(tree, "--apply", "--json")
    after = json.loads(capsys.readouterr().out)
    assert after["applied"] is True
    assert after["removed_bytes"] == before
    assert after["removed_files"] > 0
    assert after["errors"] == []
    assert not (tree / "budget-tracker" / "node_modules").exists()
    assert not (tree / "budget-tracker" / ".expo").exists()


def test_apply_keeps_everything_authored(tree):
    run(tree, "--apply")
    for name in ("budget-tracker", "workout-log"):
        app = tree / name
        assert survivors(app) == {
            "screens", "storage", "app.db", "package.json",
            "package-lock.json", ".git",
        }, survivors(app)
        assert (app / "screens" / "HomeScreen.tsx").read_text().startswith("export")
        assert json.loads((app / "storage" / "user.json").read_text())["kept"] is True
        assert (app / "app.db").is_file()
        assert (app / ".git" / "HEAD").is_file()


def test_apply_is_idempotent(tree, capsys):
    run(tree, "--apply")
    capsys.readouterr()  # drop the first run's human output
    run(tree, "--apply", "--json")
    second = json.loads(capsys.readouterr().out)
    assert second["removed_bytes"] == 0
    assert second["errors"] == []


# ── 3. Refusals ──────────────────────────────────────────────────────

def test_a_symlinked_node_modules_is_not_followed(tmp_path, monkeypatch):
    """A symlink named node_modules pointing outside the root would make this
    script an arbitrary-deletion primitive running as root."""
    monkeypatch.setattr(cleanup, "_listening_ports", lambda: set())
    monkeypatch.setattr(cleanup, "metro_tmp_caches", lambda: [])
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "secrets.env").write_text("TOKEN=1")

    root = tmp_path / "apps"
    (root / "sneaky").mkdir(parents=True)
    os.symlink(outside, root / "sneaky" / "node_modules")

    run(root, "--apply")
    assert (outside / "secrets.env").is_file()
    assert (root / "sneaky" / "node_modules").is_symlink()


def test_running_apps_are_skipped(tree, monkeypatch, capsys):
    monkeypatch.setattr(cleanup, "_listening_ports", lambda: {4001})
    run(tree, "--apply", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed_bytes"] == 0
    for a in payload["apps"]:
        assert a["skipped"], a
    assert (tree / "budget-tracker" / "node_modules").is_dir()


def test_keep_newest_protects_the_most_recent(tree, capsys):
    os.utime(tree / "workout-log", (1, 1))  # make budget-tracker the newest
    run(tree, "--keep-newest", "1", "--apply", "--json")
    payload = json.loads(capsys.readouterr().out)
    protected = [a for a in payload["apps"] if a["skipped"]]
    assert [a["app"] for a in protected] == ["budget-tracker"]
    assert (tree / "budget-tracker" / "node_modules").is_dir()
    assert not (tree / "workout-log" / "node_modules").exists()


def test_a_missing_root_is_not_an_error(tmp_path, capsys):
    assert run(tmp_path / "nope") == 0
    assert "nothing to reclaim" in capsys.readouterr().out


def test_the_html_app_root_is_never_a_target(tmp_path, monkeypatch, capsys):
    """The two roots are separate directories on purpose; a cleanup aimed at
    one must be incapable of reaching the other."""
    monkeypatch.setattr(cleanup, "_listening_ports", lambda: set())
    monkeypatch.setattr(cleanup, "metro_tmp_caches", lambda: [])
    expo_root = tmp_path / "opt" / "toup-agent" / "apps"
    expo_root.mkdir(parents=True)
    build_fake_app(expo_root, "legacy-app")

    html_root = tmp_path / "app" / "workspace" / "apps"
    html_root.mkdir(parents=True)
    (html_root / "budget-tracker.html").write_text("<!doctype html><html><body>hi</body></html>")
    (html_root / "manifest.json").write_text("{}")

    run(expo_root, "--apply", "--json")
    payload = json.loads(capsys.readouterr().out)
    assert payload["removed_bytes"] > 0  # control: it did delete something
    assert (html_root / "budget-tracker.html").is_file()
    assert (html_root / "manifest.json").is_file()
    for a in payload["apps"]:
        for target in a["targets"]:
            assert str(html_root) not in target


# ── 4. The measurement is a measurement ──────────────────────────────

def test_reported_reclaimable_equals_what_is_actually_removed(tree, capsys):
    """The number this script prints is the number an operator will quote in
    a rollout report, so it has to be measured rather than modelled.

    Note this asserts CORRECTNESS of the measurement, not a size ratio: the
    ratio in a synthetic fixture is a property of the fixture. The real
    ratio (99.7 % of an app directory is node_modules) is measured on an
    actual `create-expo-app` + `npm install` tree and recorded in
    MIGRATION_INVENTORY.md §3.
    """
    run(tree, "--json")
    predicted = json.loads(capsys.readouterr().out)

    # Independently measure the named targets before anything is deleted.
    independent = 0
    for a in predicted["apps"]:
        for target in a["targets"]:
            independent += cleanup.dir_size(target)[0]
    assert independent == predicted["reclaimable_bytes"]

    run(tree, "--apply", "--json")
    actual = json.loads(capsys.readouterr().out)
    assert actual["removed_bytes"] == predicted["reclaimable_bytes"]


def test_dir_size_counts_a_hardlink_once(tmp_path):
    """npm hardlinks from its global cache. Counting a hardlinked file per
    link would inflate every reclaim figure this script reports."""
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.js").write_text("z" * 4096)
    os.link(d / "a.js", d / "b.js")
    once, files = cleanup.dir_size(str(d))
    assert files == 1, "the hardlink was counted twice"
    assert once > 0
