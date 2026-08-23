"""Round 18 — the build tells the truth about the app it built.

Three defects, one theme. A build could report success it had not earned:

* the shell refused ordinary commands and reported it as a parse error the
  model could not act on (``could not parse the command (No closing
  quotation)`` for ``grep -E "a|b"``);
* nothing ever looked at the generated JavaScript, so an app whose script died
  on its first line was published as finished;
* the app's own words to the model — internal tool names, an internal route, a
  chip directive, stage direction — were shown to the user as the agent's.

Each section below is one of those, and each asserts against the thing that
actually runs (the segmenter, node, the served bytes) rather than against a
description of it.
"""

from __future__ import annotations

import asyncio
import os
import shutil

import pytest

from app.agent.skills.builtins.app_html import runtime, shell, store, verify
from app.agent.skills.builtins.app_html import steps as steps_mod
from app.agent.skills.builtins.app_html.skill import AppHtmlSkill
from app.agent.skills.base import SkillContext


CTX = SkillContext(workspace="/tmp", user_id="user-1", session_id="s1")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def apps_dir(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("TOUP_APP_SMOKE_TEST", "0")
    (tmp_path / "home").mkdir()
    store.ensure_root()
    return root


@pytest.fixture()
def skill(apps_dir, monkeypatch):
    s = AppHtmlSkill()
    emitted = []

    async def _noop(*_a, **_k):
        return None

    async def _job(*_a, **_k):
        return "job-test"

    async def _emit(**kw):
        emitted.append(kw)

    async def _upsert(**_kw):
        return "app-test"

    monkeypatch.setattr(steps_mod, "ensure_job", _job)
    monkeypatch.setattr(steps_mod, "emit_step", _emit)
    monkeypatch.setattr(steps_mod, "finish_job", _noop)
    monkeypatch.setattr(steps_mod, "announce_ready", _noop)
    monkeypatch.setattr(steps_mod, "upsert_app_row", _upsert)
    s._emitted = emitted  # type: ignore[attr-defined]
    return s


#: Round 20 made `brief` a required argument of `create_app_file` — an app
#: whose purpose was never written down is an app whose next edit is a guess.
#: Tests calling the tool the ordinary way get an ordinary brief; the
#: requirement itself is asserted in test_app_brief.py, which passes nothing.
SAMPLE_BRIEF = (
    "## What it is\n"
    "A one-screen arcade game for someone with a spare minute on their phone. "
    "It solves 'I want something to do for sixty seconds that does not need an "
    "account'.\n\n"
    "## Core flows\n"
    "- Press Play, steer with the D-pad, eat, score, lose.\n"
    "- Press Play again from the game-over card.\n\n"
    "## Features, states and controls\n"
    "- States: start screen, playing, over. Play starts the loop; the D-pad "
    "sets direction; the score reads live.\n\n"
    "## Design decisions\n"
    "- Near-black field with one warm accent so the board is the only bright "
    "thing; 64px D-pad keys because they are pressed continuously."
)


def _with_brief(name, args):
    if name == "create_app_file" and "brief" not in args:
        args = {**args, "brief": SAMPLE_BRIEF}
    return args

def call(sk, name, **args):
    return run(sk.execute_tool(f"app_html__{name}", _with_brief(name, args), CTX))


def snake_html(*, broken: bool = False, storage: bool = False) -> str:
    """A real single-file app, of the shape and size the pipeline produces."""
    script = """
const board = document.getElementById('board');
const ctx = board.getContext('2d');
let snake = [{x: 8, y: 8}];
let dir = {x: 1, y: 0};
let food = {x: 12, y: 8};
let score = 0;

function step() {
  const head = {x: snake[0].x + dir.x, y: snake[0].y + dir.y};
  snake.unshift(head);
  if (head.x === food.x && head.y === food.y) {
    score += 1;
    document.getElementById('score').textContent = String(score);
    food = {x: (head.x + 5) % 20, y: (head.y + 3) % 20};
  } else {
    snake.pop();
  }
  draw();
}

function draw() {
  ctx.fillStyle = '#0B0B0F';
  ctx.fillRect(0, 0, 400, 400);
  ctx.fillStyle = '#FF5C39';
  for (const s of snake) ctx.fillRect(s.x * 20, s.y * 20, 18, 18);
}

addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') dir = {x: 1, y: 0};
  if (e.key === 'ArrowLeft') dir = {x: -1, y: 0};
});
setInterval(step, 120);
draw();
"""
    if storage:
        # Unguarded, on the FIRST line — the shape that killed a page.
        script = "let best = Number(localStorage.getItem('best') || 0);\n" + script
    if broken:
        # A generation that ran out of room mid-function. Valid HTML, dead JS.
        script = script.replace("function draw() {", "function draw( {")
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
        "<title>Nokia Snake</title>\n<style>\n"
        ":root{--bg:#0B0B0F;--ink:#F4F4F5;--accent:#FF5C39}\n"
        "body{margin:0;background:var(--bg);color:var(--ink);"
        "font:16px/1.5 system-ui,sans-serif;display:grid;place-items:center;"
        "min-height:100vh}\n"
        "canvas{border:2px solid var(--accent);max-width:100%}\n"
        ".btn:focus-visible{outline:3px solid var(--ink);outline-offset:2px}\n"
        "</style>\n</head>\n<body>\n"
        "<h1>Nokia Snake</h1>\n<p>Score: <span id=\"score\">0</span></p>\n"
        "<canvas id=\"board\" width=\"400\" height=\"400\"></canvas>\n"
        f"<script>{script}</script>\n</body>\n</html>\n"
    )


# ── 1. The shell parses a shell command ───────────────────────────────

@pytest.mark.parametrize("command", [
    'grep -c "requestAnimationFrame\\|setInterval" snake.html',
    'grep -nE "gameOver|restart" snake.html',
    "awk '{print $(NF)}' snake.html",
    "grep '${VAR}' snake.html",
    'grep -o "a&b" snake.html',
    'grep -c "score" snake.html | head -1',
    "grep -c 'canvas' snake.html && wc -c snake.html",
])
def test_a_wellformed_command_is_not_refused(apps_dir, command):
    """Every one of these was refused before round 18.

    The segmenter was `re.split(r"\\|\\||&&|[;|\\n]", cmd)` — a regex over a
    string that has quotes in it. `grep -nE "gameOver|restart"` was split
    inside the pattern, and shlex was handed `grep -nE "gameOver`, which is
    an unterminated quote. The model got `could not parse the command (No
    closing quotation)` about a command with balanced quotes, twice, and then
    gave up on checking its work.
    """
    assert shell.validate_command(command) == command.strip()


@pytest.mark.parametrize("command,reason", [
    ('cat "unclosed.html', "unclosed"),
    ("echo $(whoami)", "substitution"),
    ('echo "$(whoami)"', "substitution"),   # live inside double quotes
    ("curl https://example.com", "allowed command"),
    ("sleep 60 &", "backgrounding"),
    ("cat ../../etc/passwd", "outside the app directory"),
])
def test_the_gates_still_hold(apps_dir, command, reason):
    """The control for the test above: quote-awareness must not have opened
    anything. Each of these is refused, and for the stated reason."""
    with pytest.raises(shell.ShellRefusal) as exc:
        shell.validate_command(command)
    assert reason in str(exc.value).lower()


def test_the_unclosed_quote_message_is_actionable(apps_dir):
    """When the quoting really is broken, the model is told what to do about
    it — not handed `ValueError: No closing quotation`."""
    with pytest.raises(shell.ShellRefusal) as exc:
        shell.validate_command('grep "half a pattern snake.html')
    msg = str(exc.value)
    assert "unclosed" in msg.lower()
    assert "No closing quotation" not in msg


# ── 2. The app is checked before it is published ──────────────────────

@pytest.mark.skipif(not shutil.which("node"), reason="node is the parser here")
def test_a_broken_script_is_caught_at_write_time(skill, apps_dir):
    out = call(skill, "create_app_file", slug="snake", title="Nokia Snake",
               html=snake_html(broken=True))
    assert not out.startswith("ERROR:"), out          # the file WAS written
    assert (apps_dir / "snake.html").is_file()
    assert "does not parse" in out
    # The real line in the app file, not "line 12 of the third script block".
    assert "line 3" in out or "line 2" in out or "line 4" in out, out


@pytest.mark.skipif(not shutil.which("node"), reason="node is the parser here")
def test_a_broken_app_is_not_published(skill, apps_dir):
    """The gate. Before this, `present_app` published whatever was on disk —
    which is how a card came to read "100% · App built" over a dead game."""
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html(broken=True))
    out = call(skill, "present_app", slug="snake")
    assert out.startswith("ERROR:"), out
    assert "not working yet" in out
    assert store.read_manifest()["snake"].presented_at is None


@pytest.mark.skipif(not shutil.which("node"), reason="node is the parser here")
def test_the_repair_loop_closes(skill, apps_dir):
    """A failed check is a step in the build, not the end of it: fix it, call
    again, and the app publishes. This is the whole self-heal contract."""
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html(broken=True))
    assert call(skill, "present_app", slug="snake").startswith("ERROR:")

    fixed = call(skill, "edit_app_file", slug="snake",
                 old_string="function draw( {", new_string="function draw() {",
                 reason="close the parameter list")
    assert not fixed.startswith("ERROR:"), fixed

    out = call(skill, "present_app", slug="snake")
    assert not out.startswith("ERROR:"), out
    assert store.read_manifest()["snake"].presented_at


@pytest.mark.skipif(not shutil.which("node"), reason="node is the parser here")
def test_the_checker_is_not_vacuous(apps_dir):
    """A test that cannot fail proves nothing: the same function must pass
    the good file and fail the mutated one."""
    good = run(verify.verify_app(snake_html(), deep=False))
    bad = run(verify.verify_app(snake_html(broken=True), deep=False))
    assert good.ok, [f.as_text() for f in good.findings]
    assert not bad.ok
    assert "syntax" in good.ran


def test_babel_and_json_blocks_are_not_parsed_as_javascript(apps_dir):
    """`<script type="text/babel">` is JSX and `application/json` is data.
    Handing either to node produces a SyntaxError for a perfectly good app —
    a checker that fails valid React apps is worse than no checker."""
    doc = (
        "<!doctype html><html><head><title>x</title></head><body><div id=r></div>"
        '<script type="application/json">{"a": 1}</script>'
        '<script type="text/babel">const A = () => <div className="x"/>;</script>'
        "</body></html>"
    )
    assert verify.inline_scripts(doc) == []
    assert run(verify.verify_app(doc, deep=False)).ok


# ── 3. Nothing internal reaches the user ──────────────────────────────

_INTERNAL_SHAPES = ("app_html__", "[[", "/api/", "Tell the user", "---\nname:",
                    "exit 0", "exit 1")


def test_no_tool_result_shows_the_user_anything_internal(skill, apps_dir):
    """The round-18 leak, asserted at the boundary that produces it.

    `client_summary` is the exact function `agent_runner` calls before a
    result goes on the wire, so this is what the chat renders — not a
    paraphrase of it.
    """
    from app.agent.tool_display import client_summary

    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    results = [
        call(skill, "view_app_file", slug="snake"),
        call(skill, "bash_app", slug="snake", command="wc -c snake.html"),
        call(skill, "edit_app_file", slug="snake", old_string="Score:",
             new_string="Points:", reason="rename the label"),
        call(skill, "present_app", slug="snake"),
        # A failure, too: an error's user-facing half is the half most likely
        # to carry a tool name and a next-step instruction.
        call(skill, "edit_app_file", slug="snake", old_string="nope",
             new_string="x"),
        call(skill, "bash_app", slug="snake", command="curl https://evil.example"),
    ]
    for raw in results:
        shown = client_summary(raw, cap=200)
        for shape in _INTERNAL_SHAPES:
            assert shape not in shown, (shape, shown)
        # And it is not empty: a redactor that returns "" for everything would
        # pass the loop above and tell the user nothing.
        assert len(shown) > 3, shown


def test_the_whole_app_is_not_pasted_into_the_chat(skill, apps_dir):
    """`view_app_file` returns the entire document to the model. Rendered,
    that is the user's own markup arriving as the agent's reply."""
    from app.agent.tool_display import client_summary

    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    shown = client_summary(call(skill, "view_app_file", slug="snake"), cap=200)
    assert "<!doctype" not in shown.lower()
    assert "canvas" not in shown.lower()
    assert shown == "Read the app."


def test_the_redactor_knows_the_round18_shapes():
    """The guarantee layer, independent of any tool having been converted."""
    from app.agent.tool_display import sanitize_for_client

    out = sanitize_for_client(
        "Presented 'Snake' — live at /api/artifacts/nokia-snake. Offer the "
        "[[open_app:nokia-snake]] chip. Tell the user what it does. "
        "Call app_html__present_app when ready."
    )
    assert "[[" not in out
    assert "app_html__" not in out
    assert "Tell the user" not in out

    frontmatter = sanitize_for_client(
        "---\nname: toup-frontend-design\ndescription: How a Toup app must "
        "look\n---\n\n# Toup frontend design\n\nCommit to a look."
    )
    assert "name: toup-frontend-design" not in frontmatter
    assert "Commit to a look" in frontmatter


def test_a_status_line_is_never_a_wire_identifier():
    """`app_html__create_app_file` humanises to "App html", which is what a
    lock screen showed for nine seconds before appending "— still going"."""
    from app.agent.turn_progress import _subtitle_for

    assert _subtitle_for("app_html__create_app_file") == "Building your app…"
    for name in ("app_html__bash_app", "some_future__unmapped_tool"):
        line = _subtitle_for(name)
        assert "__" not in line
        assert "app html" not in line.lower()


def test_step_labels_are_sentences_not_measurements():
    """`9299 bytes`, `exit 1`, `revision 2` and `wc -c nokia-snake.html` were
    all rendered as a step's own label. A label is not an argument any more —
    there is no code path left through which one of those can become one."""
    from app.agent.skills.builtins.app_html.steps import phase_label, initial_steps

    for step in initial_steps():
        assert step["label"][0].isupper()
        assert not any(ch.isdigit() for ch in step["label"])
    for t in ("create", "verify", "present", "edit", "review"):
        for status in ("pending", "running", "done", "failed"):
            label = phase_label(t, status)
            assert label and " " in label and "_" not in label


@pytest.mark.parametrize("message,from_console,expected", [
    # Uncaught exceptions — always a refusal.
    ("ReferenceError: draw is not defined", False, True),
    ("TypeError: Cannot read properties of null", False, True),
    # An app logging deliberately is not a broken app. Refusing over one
    # would make the gate a nuisance, and a gate people turn off is not one.
    ("Could not parse that amount — try 12.50", True, False),
    ("Warning: each child in a list should have a unique key", True, False),
    # …but a swallowed exception named in a console error still is.
    ("Uncaught TypeError: render is not a function", True, True),
    ("drawFrame is not defined", True, True),
    # A library that never arrived is this container's problem, not the
    # model's. On a box with no egress every React app would otherwise fail.
    ("Failed to load resource: net::ERR_NAME_NOT_RESOLVED", True, False),
    ("the server responded with a status of 404", True, False),
    ("", False, False),
])
def test_only_real_breakage_stops_a_publish(message, from_console, expected):
    assert verify.counts_as_breakage(message, from_console=from_console) is expected


# ── 4. The runtime the browser actually gets ──────────────────────────

def test_the_served_document_cannot_be_killed_by_storage(apps_dir):
    served = runtime.wrap_for_runtime(snake_html(storage=True))
    assert served.index(runtime.MARKER) < served.index("localStorage.getItem")
    assert "defineProperty" in served


def test_cdnjs_scripts_are_marked_crossorigin(apps_dir):
    """Without this attribute a library's exception is reported as the string
    "Script error." with no file, no line and no stack. That is the opaque
    message users were shown, and it is a two-word fix on the tag."""
    doc = (
        "<!doctype html><html><head><title>x</title>"
        '<script src="https://cdnjs.cloudflare.com/ajax/libs/react/18.2.0/react.js"></script>'
        "</head><body>hi</body></html>"
    )
    out = runtime.wrap_for_runtime(doc)
    assert 'crossorigin="anonymous"' in out
    # Idempotent — an already-marked tag is not marked twice, and a document
    # that has been through the wrapper is not wrapped again.
    assert runtime.wrap_for_runtime(out) == out
    assert out.count("crossorigin") == 1


def test_the_error_hooks_are_installed_before_the_app_runs(apps_dir):
    served = runtime.wrap_for_runtime(snake_html())
    assert served.index("unhandledrejection") < served.index("const board")
    assert runtime.ERROR_SOURCE in served


def test_the_file_on_disk_is_never_touched(skill, apps_dir):
    """The wrap happens at SERVE time. If it happened at write time,
    `view_app_file` would show the user's app with our shim in it,
    `edit_app_file`'s exact-string match would drift, and the byte count on
    the card would be wrong."""
    original = snake_html()
    call(skill, "create_app_file", slug="snake", title="Nokia Snake", html=original)
    on_disk = (apps_dir / "snake.html").read_text()
    assert on_disk == original
    assert runtime.MARKER not in on_disk
    assert store.read_manifest()["snake"].size_bytes == len(original.encode())


# ── 5. Ten builds from a clean workspace ──────────────────────────────

def test_ten_consecutive_builds_from_a_clean_state(skill, apps_dir):
    """The acceptance run. Ten apps, each written, inspected, edited and
    published, with the workspace starting empty.

    Every phase asserts on its own outcome rather than on the last one — a
    loop that only checks the final result passes with nine failures in it.
    """
    for i in range(10):
        slug = f"snake-{i}"
        created = call(skill, "create_app_file", slug=slug, title=f"Snake {i}",
                       html=snake_html())
        assert not created.startswith("ERROR:"), (slug, created)
        assert "does not parse" not in created, (slug, created)

        checked = call(skill, "bash_app", slug=slug,
                       command=f'grep -cE "setInterval|requestAnimationFrame" {slug}.html')
        assert not checked.startswith("ERROR:"), (slug, checked)
        assert checked.startswith("exit 0"), (slug, checked)

        edited = call(skill, "edit_app_file", slug=slug,
                      old_string="<h1>Nokia Snake</h1>",
                      new_string=f"<h1>Nokia Snake {i}</h1>",
                      reason="number the title")
        assert not edited.startswith("ERROR:"), (slug, edited)

        shown = call(skill, "present_app", slug=slug)
        assert not shown.startswith("ERROR:"), (slug, shown)
        assert store.read_manifest()[slug].presented_at, slug

    assert len(store.read_manifest()) == 10
    for i in range(10):
        assert (apps_dir / f"snake-{i}.html").is_file()

    # No step anywhere in the ten builds reported a failure.
    failed = [e for e in skill._emitted if e.get("status") == "failed"]
    assert failed == [], failed


# ── 6. Forced failures self-heal ──────────────────────────────────────

def test_an_unwritable_workspace_is_repaired_not_reported(skill, apps_dir):
    """The `chmod`, not `chown`, lesson at its smallest scale: a directory
    this process cannot write is a directory whose mode it can fix. Round 15
    lost a whole release to the other version of this."""
    call(skill, "create_app_file", slug="first", title="First", html=snake_html())
    os.chmod(apps_dir, 0o500)
    try:
        out = call(skill, "create_app_file", slug="second", title="Second",
                   html=snake_html())
    finally:
        os.chmod(apps_dir, 0o755)
    assert not out.startswith("ERROR:"), out
    assert (apps_dir / "second.html").is_file()


def test_an_unwritable_workspace_that_cannot_be_repaired_says_so_plainly(
    skill, apps_dir, monkeypatch,
):
    """The control for the test above. When the repair genuinely cannot work,
    the failure is one sentence with no errno, no uid and no path in it —
    because it lands in a step detail under a progress bar."""
    monkeypatch.setattr(store, "repair_permissions", lambda _p: False)
    os.chmod(apps_dir, 0o500)
    try:
        out = call(skill, "create_app_file", slug="third", title="Third",
                   html=snake_html())
    finally:
        os.chmod(apps_dir, 0o755)
    assert out.startswith("ERROR:")
    assert "can't be written to" in out
    for machinery in ("EACCES", "errno", "Permission denied", str(apps_dir)):
        assert machinery not in out


def test_a_vanished_file_is_restored_from_its_last_revision(skill, apps_dir):
    """A manifest row with no file is a missing FILENAME, not a missing app —
    the last kept revision is right there. Telling the user their app does
    not exist while a copy of it sits in `.versions/` is the wrong answer."""
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    # A second write creates the snapshot the restore reads.
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html().replace("Nokia Snake", "Nokia Snake II"))
    os.unlink(apps_dir / "snake.html")

    out = call(skill, "view_app_file", slug="snake")
    assert not out.startswith("ERROR:"), out
    assert (apps_dir / "snake.html").is_file()


def test_a_refusal_still_refuses_when_there_is_nothing_to_restore(skill, apps_dir):
    """Control: the restore must not invent an app that was never built."""
    out = call(skill, "view_app_file", slug="never-existed")
    assert out.startswith("ERROR:")
    assert not (apps_dir / "never-existed.html").exists()


# ── 7. Where the slug lives, now that the prose does not carry it ─────

def test_the_slug_is_persisted_as_a_field():
    """`present_app`'s result no longer contains `/api/artifacts/<slug>` or
    `[[open_app:<slug>]]`, which is how the clients used to recover it. That
    is only safe because it now rides `app_artifact` on the message."""
    import json
    from types import SimpleNamespace
    from app.api.day_chats import _serialize_app_artifact

    msg = SimpleNamespace(metadata_json=json.dumps(
        {"app_artifact": {"slug": "nokia-snake"}, "tool_events": []}))
    assert _serialize_app_artifact(msg) == {"slug": "nokia-snake"}

    # Malformed, absent and slugless all answer None rather than raising —
    # this runs on every message of every history load.
    for raw in (None, "", "not json", "{}", '{"app_artifact": {}}',
                '{"app_artifact": "nokia-snake"}'):
        assert _serialize_app_artifact(SimpleNamespace(metadata_json=raw)) is None


def test_every_message_reader_projects_the_slug():
    """The round-16 lesson, applied before it costs anything: a marker row
    leaked because ONE of four readers had the projection. The clients fall
    back between these endpoints, so a field carried by three of them
    disappears the moment the primary fails — which is when a fallback is
    for.
    """
    import pathlib
    repo = pathlib.Path(__file__).resolve().parents[1]
    readers = {
        "app/api/day_chats.py": 2,        # the by-date and by-day-chat paths
        "app/api/messages_recover.py": 1,
        "app/api/sessions.py": 1,
    }
    for path, expected in readers.items():
        body = (repo / path).read_text()
        got = body.count("app_artifact=") + body.count('"app_artifact":')
        assert got >= expected, f"{path} projects app_artifact {got}×, want {expected}"
    # And the response model must DECLARE it, or pydantic drops the key on
    # the session route regardless of what the converter passes.
    assert "app_artifact" in (repo / "app/schemas.py").read_text()


# ── 8. The step model ─────────────────────────────────────────────────

def test_a_failed_grep_is_not_a_failed_step(skill, apps_dir):
    """`grep` exiting 1 means it found nothing, which is an answer. The step
    used to be marked FAILED for it — a red ✗ on a check that worked, inside
    a card that also said the build had succeeded."""
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    skill._emitted.clear()
    out = call(skill, "bash_app", slug="snake",
               command="grep -c 'definitely-not-in-this-file' snake.html")
    assert out.startswith("exit 1")
    assert [e["status"] for e in skill._emitted] == ["running", "done"]


def test_a_failed_phase_says_what_actually_happened(apps_dir):
    """`BuildJob.user_message` is the only error text the clients render, and
    this pipeline wrote none — so a failed build fell through to the mobile
    card's fallback, "The build stopped partway. Nothing was changed."

    For a build whose file had already been written that is false, and false
    in the direction that matters: it tells someone their work is gone when
    it is on disk and one edit from working.
    """
    from app.agent.skills.builtins.app_html.steps import (
        _ERROR_CLASS, _PHASE_USER_MESSAGE, STEP_TYPES,
    )
    assert set(_PHASE_USER_MESSAGE) == set(STEP_TYPES)
    for phase, msg in _PHASE_USER_MESSAGE.items():
        assert "nothing was changed" not in msg.lower(), phase
        assert msg[0].isupper() and msg.rstrip().endswith("."), phase
        for machinery in ("exit ", "app_html", "ERROR:", "/app/", "None"):
            assert machinery not in msg, (phase, machinery)
    # A closed-enum value, or every reader gets a routing key it does not
    # route (job_status.py owns the list).
    from app.agent import job_status
    assert _ERROR_CLASS == job_status.ERR_TOOL_FAILURE


def test_no_step_detail_is_a_command_or_an_exit_code(skill, apps_dir):
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    call(skill, "bash_app", slug="snake", command="wc -c snake.html")
    call(skill, "present_app", slug="snake")
    for e in skill._emitted:
        detail = e.get("detail") or ""
        assert "wc -c" not in detail
        assert not detail.startswith("exit ")
        assert "app_html__" not in detail


# ── 6. An app that is over before anyone played it ────────────────────
#
# The build that prompted this section reported 100%, opened, rendered
# perfectly, and was on GAME OVER — score 000 — 1.75 s after load, because
# `reset()` and `setInterval` were the last line of the file. Every existing
# gate passed it: nothing threw, no console error, the document parsed. The
# only thing wrong was that the clock was already running when the sheet
# opened, and that is a question no error handler can be asked.


class _FakePage:
    """A page whose visible text follows a script of (elapsed_ms, text)."""

    def __init__(self, timeline):
        self._timeline = sorted(timeline)
        self._now = 0

    async def evaluate(self, _js):
        text = ""
        for at, value in self._timeline:
            if at <= self._now:
                text = value
        return text

    async def wait_for_timeout(self, ms):
        self._now += ms


def _watch(timeline, first_text):
    """Run the real watcher over a scripted page, from the settle onward."""
    page = _FakePage(timeline)
    page._now = verify.SMOKE_SETTLE_MS
    return run(verify._ended_itself(page, first_text))


def test_a_game_that_ends_itself_untouched_is_a_finding():
    # Alive at first paint, over before the watch window closes.
    assert _watch([(0, "SCORE 000  BEST 000"),
                   (1800, "GAME OVER\nTHE SNAKE HIT THE WALL.")],
                  first_text="SCORE 000  BEST 000") == "GAME OVER"


def test_a_slow_self_end_is_still_caught_within_the_window():
    assert _watch([(0, "TIME 30"), (2900, "TIME'S UP")],
                  first_text="TIME 30") == "TIME'S UP"


def test_an_app_that_stays_alive_is_not_a_finding():
    assert _watch([(0, "SCORE 000"), (2900, "SCORE 040")],
                  first_text="SCORE 000") is None


def test_an_end_state_already_on_screen_at_first_paint_is_not_a_finding():
    # A leaderboard is entitled to the words "Final score" at rest. The
    # defect is a verdict that ARRIVES; the baseline is what separates them.
    assert _watch([(0, "Final score 120"), (2000, "Final score 120")],
                  first_text="Final score 120") is None


@pytest.mark.parametrize("text,terminal", [
    ("GAME OVER", True),
    ("Game Over — try again", True),
    ("You lose!", True),
    ("Time's up", True),
    ("Final score", True),
    # Ordinary states a healthy app shows at rest.
    ("Score 40  Lives 3", False),
    ("Round 2 of 5", False),
    ("Time left 0:42", False),
    ("Best score", False),
    ("Start game", False),
])
def test_only_a_terminal_verdict_counts(text, terminal):
    assert bool(verify._TERMINAL_TEXT_RE.search(text)) is terminal


def test_the_watch_outlasts_the_settle():
    # The check exists to see PAST the frame a healthy app has finished
    # painting on. Equal values would make it a second reading of the same
    # instant, and the defect it catches lands after that instant.
    assert verify.SELF_END_WATCH_MS > verify.SMOKE_SETTLE_MS


def test_the_finding_is_worded_as_something_to_fix_and_names_no_internals():
    from app.agent.skills.builtins.app_html import verify as v
    msg = (
        f"the app reached “GAME OVER” on its own, "
        f"{v.SELF_END_WATCH_MS // 1000}s after opening, with nothing "
        "pressed — so the user sees it already over. Do not start a "
        "clock, a loop or a countdown at load: open on a start "
        "screen and begin in the start control's handler."
    )
    # It has to survive the same gate every other browser message goes
    # through, or it is collected and silently dropped.
    assert v.counts_as_breakage(msg, from_console=False) is True
    for machinery in ("app_html", "ERROR:", "/app/", "setInterval("):
        assert machinery not in msg


# ── 7. Which build a turn was, after the app is relaunched ────────────
#
# The build card is drawn from a job id, and the job id reached the client
# only on `job_update` frames — a live-only channel. So a turn that showed
# one card while it ran came back, on the next launch, as the generic
# "N actions" rail that this round exists to remove. Measured on a
# simulator: correct live, degraded on reload.
#
# The fix is the same one `app_slug` already got one commit earlier — a
# FIELD on the tool record — so these assert the two places a field like
# that gets silently lost: the derivation, and the persistence allowlist.


def test_the_build_job_id_survives_the_ingest_allowlist():
    # `_clean_tool_events` drops every key not named here, so a field the
    # runner stamps and this set omits reaches chat and not history — which
    # is indistinguishable, from the client, from never having been sent.
    from app.api.sessions import _TOOL_EVENT_KEYS
    assert "job_id" in _TOOL_EVENT_KEYS
    assert "app_slug" in _TOOL_EVENT_KEYS


def test_a_cleaned_tool_event_keeps_the_build_it_belongs_to():
    from app.api.sessions import _clean_tool_events
    out = _clean_tool_events([{
        "tool": "app_html__present_app",
        "call_id": "call_1",
        "started_at_ms": 1,
        "completed_at_ms": 2,
        "summary": "Tip & Split is ready.",
        "label": "Finishing your app",
        "app_slug": "tip-split",
        "job_id": "c028d8f4-835b-4178-8279-d3c49eb93d8e",
    }])
    assert out and out[0]["job_id"] == "c028d8f4-835b-4178-8279-d3c49eb93d8e"
    assert out[0]["app_slug"] == "tip-split"


def test_the_job_id_is_derived_from_the_slug_the_pipeline_owns():
    # The client must never recompute this: `_APP_NS` and the `user:slug` key
    # are steps.py's private business, and a client copy would break silently
    # the day either changed. So the derivation is asserted HERE.
    from app.agent.skills.builtins.app_html import steps as s
    a = s.app_id_for("user-1", "tip-split")
    assert a == s.app_id_for("user-1", "tip-split")        # stable
    assert a != s.app_id_for("user-2", "tip-split")        # per user
    assert a != s.app_id_for("user-1", "tip-splits")       # per slug
    assert s.job_id_for_slug.__doc__ and "client" in s.job_id_for_slug.__doc__


# ── 8. The gate has to PLAY the app, not just open it ─────────────────
#
# Second pass, and a consequence of section 6. Once "anything that can be
# lost opens on a start screen" became the rule, an app's first real code
# path moved behind a gesture — and the gate's one frame of input was a
# keypress plus a click at a FIXED POINT, which does not reliably land on a
# button whose position is the app's own business.
#
# The build that showed it: a Snake whose first render called
# `classList.add('snake', '')`. An empty token is a DOMException in every
# browser. `node --check` parses it, the page loads clean, and the gate said
# "opened it — no errors" because `render()` had never run. It threw the
# instant the user pressed PLAY, on a real device.
#
# A gate that got weaker the day apps got lazier is worse than no gate.


@pytest.mark.parametrize("label,is_start", [
    ("PLAY", True),
    ("Play again", True),
    ("START", True),
    ("Start game", True),
    ("Begin", True),
    ("New game", True),
    ("GO!", True),
    # Anchored at the start, so an ordinary control is not mistaken for one.
    ("Display settings", False),
    ("Restart tour", False),
    ("Replay the intro", False),
    ("Stop", False),
    ("Cancel", False),
    ("", False),
])
def test_only_a_start_control_is_pressed(label, is_start):
    import re as _re
    assert bool(_re.search(verify._START_CONTROL_RE, label, _re.I)) is is_start


def test_the_press_script_skips_a_control_with_no_box():
    # A start screen's button is often one of several in the DOM — the pause
    # button, the game-over panel's "Play again" — with only one of them
    # laid out. Pressing a zero-box element is a no-op that looks like a
    # press, so the script must move on to the next candidate.
    assert "getBoundingClientRect" in verify._PRESS_START_JS
    assert "box.width" in verify._PRESS_START_JS and "box.height" in verify._PRESS_START_JS
    # A real click on the ELEMENT, never a coordinate — that was the defect.
    assert "el.click()" in verify._PRESS_START_JS


def test_an_empty_class_token_would_be_counted_as_breakage():
    # What the two engines actually say. Both must stop a publish, or the
    # gate is only as good as the browser the build container happens to run.
    chromium = ("Failed to execute 'add' on 'DOMTokenList': "
                "The token provided must not be empty.")
    webkit = "SyntaxError: The string did not match the expected pattern."
    for message in (chromium, webkit):
        assert verify.counts_as_breakage(message, from_console=False) is True


def test_the_app_is_driven_before_the_report_is_believed():
    # ORDER, not presence. `_press_start` has to run inside the browser
    # block and before it closes, or the errors it provokes are collected
    # after the handlers are gone. (The guard class CLAUDE.md records: a
    # check whose precondition something above it destroys.)
    import inspect
    src = inspect.getsource(verify._smoke)
    i_start = src.index("_press_start(page)")
    i_close = src.index("browser.close()")
    i_settle = src.index("SMOKE_SETTLE_MS")
    assert i_settle < i_start < i_close
    # …and AFTER the untouched watch, whose whole claim is "nothing pressed".
    assert src.index("_ended_itself") < i_start


# ── 9. A published app has to be READABLE by the shell that checks it ──
#
# `bash_app` is the model's only way to look at what it wrote, and it runs
# DROPPED to an unprivileged uid (`sandbox_preexec`) while the skill writes
# as root. `tempfile.mkstemp` creates its file 0600 and `os.replace` carries
# that mode onto the published file — so every `bash_app` check on the app
# was "Permission denied", for `wc -c` as much as for `grep`.
#
# Seen end to end on a device: asked to grep its own app, the agent answered
# "the app file is permission-blocked in the app sandbox (Permission
# denied)". The directory had been given this exact care and the file had
# not: a readable directory full of unreadable files.


def test_a_written_app_is_readable_by_a_different_uid(skill, apps_dir):
    import stat
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    mode = stat.S_IMODE((apps_dir / "snake.html").stat().st_mode)
    assert mode & stat.S_IRGRP, oct(mode)
    assert mode & stat.S_IROTH, oct(mode)
    # …and never writable by anyone but the owner.
    assert not (mode & stat.S_IWGRP) and not (mode & stat.S_IWOTH), oct(mode)


def test_the_manifest_is_readable_too(skill, apps_dir):
    # Same writer, same trap: a manifest only root can read makes the store
    # unlistable from the shell for exactly the same reason.
    import stat
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    manifest = apps_dir / "apps.json"
    if manifest.exists():
        mode = stat.S_IMODE(manifest.stat().st_mode)
        assert mode & stat.S_IROTH, oct(mode)


def test_a_rewrite_keeps_the_mode(skill, apps_dir):
    # The mode is set on the TEMP file before `os.replace`, so it has to
    # survive a second revision — a fix applied only to first creation would
    # look right and regress on the first edit.
    import stat
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html())
    call(skill, "create_app_file", slug="snake", title="Nokia Snake",
         html=snake_html(storage=True))
    mode = stat.S_IMODE((apps_dir / "snake.html").stat().st_mode)
    assert mode & stat.S_IROTH, oct(mode)


# ── Round 23b: a downgraded gate says WHY, on the card ──────────────────────
#
# The fleet ran for days with a stale chromium mount (/opt/toup/playwright at
# revision 1223, patchright wanting 1228). Every verify degraded to the syntax
# check and every card said "the code checks out" — honest, but silent about
# the gate being down. The reason now rides the step detail.

def test_a_downgraded_summary_names_its_reason():
    r = verify.Report()
    r.ran.append("syntax")
    r.downgrade_reason = "chromium missing or stale on this host"
    assert r.summary() == (
        "the code checks out — couldn't open it "
        "(chromium missing or stale on this host)"
    )
    # Without a reason the old wording stands untouched.
    r2 = verify.Report()
    r2.ran.append("syntax")
    assert r2.summary() == "the code checks out"
    # A run that DID open the browser never mentions a reason.
    r3 = verify.Report()
    r3.ran.extend(["syntax", "runtime"])
    r3.downgrade_reason = "leftover"
    assert r3.summary() == "opened it — no errors"


def test_downgrade_reason_classifies_the_two_known_infra_faults():
    stale = Exception(
        "BrowserType.launch: Executable doesn't exist at "
        "/root/.cache/ms-playwright/chromium_headless_shell-1228/headless_shell"
    )
    assert verify._downgrade_reason_of(stale) == "chromium missing or stale on this host"
    assert verify._downgrade_reason_of(ImportError("no playwright")) == "playwright not installed"
    long = Exception("x" * 200)
    assert len(verify._downgrade_reason_of(long)) <= 60


def test_smoke_failure_carries_the_reason_into_the_report(monkeypatch):
    async def boom(html, report):
        raise Exception(
            "BrowserType.launch: Executable doesn't exist at /nope/chromium-1228"
        )
    monkeypatch.setattr(verify, "_smoke", boom)
    rep = asyncio.get_event_loop().run_until_complete(
        verify.smoke_test("<html></html>")
    )
    assert "runtime" in rep.skipped and "runtime" not in rep.ran
    assert rep.downgrade_reason == "chromium missing or stale on this host"
