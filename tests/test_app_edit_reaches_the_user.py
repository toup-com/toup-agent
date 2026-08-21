"""Round 19 — the edit that never arrived, and the file nobody could read.

Three defects off one device session, all of the same shape: a step of the
pipeline reported a result it had not established.

* ``edit_app_file`` answered "revision 2, republished" and the runner kept
  showing revision 1. Not caching, and not the write path — the file on disk
  WAS revision 2 and ``/api/artifacts/{slug}`` WOULD have served it. Nothing
  ever told a client a revision 2 existed. ``announce_ready`` broadcast only
  ``app_ready`` (no revision) and the persisted metadata was ``{"slug": …}``,
  while both clients decide staleness by comparing a revision they were never
  sent. §2 and §3 below.
* ``bash_app`` answered "Permission denied" for ``grep`` and ``wc -c``. The
  write-time mode fix landed one commit earlier and repairs exactly the files
  written after it — every app already on a tenant's volume was still 0600.
  §1.
* an edit reported success on the strength of a syscall not raising, having
  never looked at the file afterwards. §4.

Every assertion here is against the thing that runs — the real sweep, the real
broadcast, the real skill dispatch — never against a description of it.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat

import pytest

from app.agent.skills.base import SkillContext
from app.agent.skills.builtins.app_html import runtime, store
from app.agent.skills.builtins.app_html import steps as steps_mod
from app.agent.skills.builtins.app_html.skill import AppHtmlSkill

CTX = SkillContext(workspace="/tmp", user_id="user-1", session_id="s1")


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


APP = """<!doctype html><html><head><meta charset="utf-8">
<title>Snake</title><style>
.key{width:44px;height:44px}
</style></head><body>
<div id="board"></div>
<div class="pad"><button class="key" data-dir="up">^</button></div>
<script>
const dirs = {up:[0,-1], down:[0,1], left:[-1,0], right:[1,0]};
let running = false;
function turn(d){ if (dirs[d]) running = true; }
document.querySelectorAll('[data-dir]').forEach(function (b) {
  b.onclick = function () { turn(b.dataset.dir); };
});
</script></body></html>
"""


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
    """The real skill with the reporting layer stubbed — every tool below runs
    its real body; only the job card and the WS are replaced."""
    s = AppHtmlSkill()

    async def _noop(*_a, **_k):
        return None

    async def _job(*_a, **_k):
        return "job-test"

    async def _upsert(**_kw):
        return "app-test"

    monkeypatch.setattr(steps_mod, "ensure_job", _job)
    monkeypatch.setattr(steps_mod, "emit_step", _noop)
    monkeypatch.setattr(steps_mod, "finish_job", _noop)
    monkeypatch.setattr(steps_mod, "announce_ready", _noop)
    monkeypatch.setattr(steps_mod, "upsert_app_row", _upsert)
    return s


def _create(skill, html=APP, slug="snake"):
    return run(skill.execute_tool(
        "app_html__create_app_file",
        {"slug": slug, "title": "Snake", "html": html}, CTX,
    ))


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


# ═════════════════════════════════════════════════════════════════════
# 1. The verification shell could not read the apps that were already there
# ═════════════════════════════════════════════════════════════════════
#
# `_atomic_write` chmods the TEMP file, so every app written from that commit
# onward is 0644 — and every app written before it is still 0600, forever. On
# an upgraded container the model can grep the app it built today and gets
# "Permission denied" for the one it built last week, which reads like a fault
# in that particular app. The repair has to be a sweep.

def test_a_file_written_before_the_fix_is_repaired(skill, apps_dir):
    _create(skill)
    path = apps_dir / "snake.html"
    os.chmod(path, 0o600)                       # the state of every legacy volume
    assert not (_mode(path) & stat.S_IROTH)

    moved = store.repair_file_modes()

    assert moved >= 1
    assert _mode(path) & stat.S_IRGRP, oct(_mode(path))
    assert _mode(path) & stat.S_IROTH, oct(_mode(path))


def test_the_sweep_reaches_the_versions_and_the_state(skill, apps_dir):
    """`shutil.copy2` carries the SOURCE's mode onto a snapshot, so a legacy
    app's history is legacy-moded too — and `diff`ing a revision is a real
    verification command."""
    _create(skill)
    run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "let running = false;",
        "new_string": "let running = false; // v2",
    }, CTX))
    store.write_state("snake", {"best": "10"})

    stale = []
    for directory in (apps_dir / ".versions" / "snake", apps_dir / ".state"):
        for name in os.listdir(directory):
            p = directory / name
            os.chmod(p, 0o600)
            stale.append(p)
    assert stale, "fixture produced neither a snapshot nor a state file"

    store.repair_file_modes()

    for p in stale:
        assert _mode(p) & stat.S_IROTH, f"{p}: {oct(_mode(p))}"


def test_the_sweep_grants_read_and_nothing_else(skill, apps_dir):
    """The brief's constraint, asserted rather than assumed.

    The dropped uid needs to LOOK at what it is verifying. Writable-by-other
    would hand the sandboxed shell — and anything else running as that uid —
    the ability to rewrite a published app behind the store's back, which is a
    new privilege, not a repaired one. Kills the `_FILE_MODE = 0o666` mutation.
    """
    _create(skill)
    os.chmod(apps_dir / "snake.html", 0o600)
    store.repair_file_modes()

    for name in os.listdir(apps_dir):
        path = apps_dir / name
        if not path.is_file():
            continue
        mode = _mode(path)
        assert not (mode & stat.S_IWGRP), f"{name} group-writable: {oct(mode)}"
        assert not (mode & stat.S_IWOTH), f"{name} world-writable: {oct(mode)}"
        assert not (mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)), (
            f"{name} executable: {oct(mode)}"
        )


def test_the_sweep_leaves_directories_alone(skill, apps_dir):
    """A directory needs +x to be entered at all; 0644 on `.versions` would
    make the whole history unreachable — a repair that breaks what it touches."""
    _create(skill)
    before = _mode(apps_dir / ".versions")
    store.repair_file_modes()
    assert _mode(apps_dir / ".versions") == before
    assert before & stat.S_IXUSR


def test_bash_app_repairs_before_it_looks(skill, apps_dir):
    """The end of the trail: the tool the model actually calls.

    `bash_app` is the one tool here that reaches a container which may never
    create another app, so it cannot depend on `ensure_root` having run since
    the upgrade. Kills the mutation that drops `repair_file_modes` from
    `run_in_app_dir`.
    """
    _create(skill)
    path = apps_dir / "snake.html"
    os.chmod(path, 0o600)

    out = run(skill.execute_tool(
        "app_html__bash_app", {"slug": "snake", "command": "wc -c snake.html"}, CTX,
    ))

    assert _mode(path) & stat.S_IROTH, oct(_mode(path))
    assert "exit 0" in str(out)


def test_the_sweep_is_idempotent_and_silent_the_second_time(skill, apps_dir):
    _create(skill)
    store.repair_file_modes()
    assert store.repair_file_modes() == 0


# ═════════════════════════════════════════════════════════════════════
# 2. Nothing ever told a client which revision was live
# ═════════════════════════════════════════════════════════════════════

@pytest.fixture()
def frames(monkeypatch):
    sent = []

    async def _capture(user_id, payload):
        sent.append(payload)

    monkeypatch.setattr(steps_mod, "_broadcast", _capture)
    return sent


#: The real one, captured before any fixture stubs the module attribute — the
#: `skill` fixture replaces `announce_ready` with a no-op so the tools can run
#: without a WS, and these tests are about `announce_ready` itself.
_REAL_ANNOUNCE = steps_mod.announce_ready


def _announce(slug="snake"):
    run(_REAL_ANNOUNCE(
        user_id="user-1", job_id="job-test", app_id="app-test",
        title="Snake", slug=slug,
    ))


def test_present_announces_which_revision_is_live(skill, apps_dir, frames):
    _create(skill)
    _announce()

    artifact = [f for f in frames if f.get("type") == "app_artifact"]
    assert artifact, f"no app_artifact frame in {[f.get('type') for f in frames]}"
    assert artifact[0]["artifact"]["revision"] == 1
    assert artifact[0]["artifact"]["slug"] == "snake"
    assert artifact[0]["artifact"]["title"] == "Snake"


def test_an_edit_moves_the_revision_the_frame_carries(skill, apps_dir, frames):
    """The defect, end to end. Two publishes of the same slug either differ by
    a revision the client can compare, or the second one is invisible."""
    _create(skill)
    _announce()
    run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "width:44px;height:44px",
        "new_string": "width:76px;height:76px",
    }, CTX))
    _announce()

    revisions = [f["artifact"]["revision"] for f in frames
                 if f.get("type") == "app_artifact"]
    assert len(revisions) == 2
    assert revisions[1] > revisions[0], revisions


def test_the_artifact_frame_arrives_before_the_card(skill, apps_dir, frames):
    """Order is load-bearing: `app_ready` is what makes a client DRAW the card,
    and a card drawn before the registry moved renders the previous revision
    and corrects itself a frame later."""
    _create(skill)
    _announce()
    kinds = [f.get("type") for f in frames]
    assert kinds.index("app_artifact") < kinds.index("app_ready")


def test_the_frame_carries_no_body(skill, apps_dir, frames):
    """A chat frame is not a delivery mechanism for 40 KB of HTML — and the
    clients fetch the body at open time by design. What has to travel is the
    number that tells them the fetch is worth making."""
    _create(skill)
    _announce()
    payload = [f for f in frames if f.get("type") == "app_artifact"][0]
    assert "html" not in payload["artifact"]
    assert len(json.dumps(payload)) < 1000


def test_an_unknown_slug_still_announces_rather_than_raising(apps_dir, frames):
    """announce_ready is fail-open: a manifest that cannot answer must cost the
    revision, never the card."""
    _announce("ghost")
    assert [f for f in frames if f.get("type") == "app_ready"]
    assert [f for f in frames if f.get("type") == "app_artifact"][0]["artifact"] == {
        "slug": "ghost",
    }


def test_the_whole_trip_end_to_end(skill, apps_dir, frames, monkeypatch):
    """Build, edit, publish — and check the three places the new revision has
    to show up agree with each other.

    The pieces are covered above; this is the walk-through, because the defect
    was never in a piece. Every individual step was correct: the write landed,
    the manifest moved, the serving route would have answered with the new
    bytes. What was missing was the join between them.
    """
    from app.api.artifacts import get_artifact

    monkeypatch.setattr(steps_mod, "announce_ready", _REAL_ANNOUNCE)

    _create(skill)
    run(skill.execute_tool("app_html__present_app", {"slug": "snake"}, CTX))
    run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "width:44px;height:44px",
        "new_string": "width:76px;height:76px",
    }, CTX))
    run(skill.execute_tool("app_html__present_app", {"slug": "snake"}, CTX))

    told = [f["artifact"]["revision"] for f in frames
            if f.get("type") == "app_artifact"]
    served = run(get_artifact("snake"))

    # 1. The client was told a NEW number …
    assert told == [1, 2], told
    # 2. … the same number the serving route reports …
    assert served.headers["X-Toup-Artifact-Revision"] == str(told[-1])
    # 3. … and the bytes behind it are the edited ones.
    assert b"width:76px" in served.body
    assert b"width:44px" not in served.body


# ═════════════════════════════════════════════════════════════════════
# 3. …and history could not correct it either
# ═════════════════════════════════════════════════════════════════════

def test_the_persisted_handle_carries_the_revision(skill, apps_dir):
    """`agent_runner` stamps this payload on the assistant message. With the
    slug alone, a client hydrating the thread starts at revision 0 for an app
    on revision 4, and every staleness check downstream compares against that
    zero."""
    _create(skill)
    run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "let running = false;",
        "new_string": "let running = true;",
    }, CTX))

    payload = steps_mod.artifact_payload("snake")

    assert payload["slug"] == "snake"
    assert payload["revision"] == 2
    assert payload["title"] == "Snake"
    assert payload["updated_at"]


def test_the_message_serializer_passes_the_whole_handle_through(skill, apps_dir):
    """The four-serializer rule: a field carried by one reader and not the
    others disappears the moment a client takes its fallback path. This asserts
    the shape is not narrowed on the way out."""
    from app.api.day_chats import _serialize_app_artifact

    _create(skill)
    payload = steps_mod.artifact_payload("snake")

    class _Msg:
        metadata_json = json.dumps({"app_artifact": payload})

    assert _serialize_app_artifact(_Msg())["revision"] == payload["revision"]


def test_the_source_route_returns_the_current_bytes_unwrapped(skill, apps_dir):
    """Where a client goes once it has been told the revision moved.

    Unwrapped on purpose: `/{slug}` returns the RUNTIME-WRAPPED document, which
    is right for a browser frame and wrong for a client that applies its own
    sandbox wrapper — it would install two storage shims over each other.
    """
    from app.api.artifacts import get_artifact_source

    _create(skill)
    run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "width:44px;height:44px",
        "new_string": "width:76px;height:76px",
    }, CTX))

    out = run(get_artifact_source("snake"))

    assert out["revision"] == 2
    assert "width:76px" in out["html"]
    assert "width:44px" not in out["html"]
    # The body is what the model wrote, byte for byte — no injected shim.
    assert out["html"] == store.read_app("snake")
    assert runtime.MARKER not in out["html"]
    assert runtime.MARKER in runtime.wrap_for_runtime(out["html"])


def test_the_source_route_404s_for_an_app_that_is_not_there(apps_dir):
    from fastapi import HTTPException

    from app.api.artifacts import get_artifact_source

    with pytest.raises(HTTPException) as exc:
        run(get_artifact_source("ghost"))
    assert exc.value.status_code == 404


# ═════════════════════════════════════════════════════════════════════
# 4. An edit reported success it had not established
# ═════════════════════════════════════════════════════════════════════

def test_an_edit_reads_the_file_back(skill, apps_dir):
    _create(skill)
    out = str(run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "width:44px;height:44px",
        "new_string": "width:76px;height:76px",
    }, CTX)))
    assert "Read back from disk" in out
    assert "the new text is in the file" in out


def test_an_edit_that_did_not_land_is_refused(skill, apps_dir, monkeypatch):
    """The claim an edit used to make was about a syscall; the claim the model
    then makes to the user is about the file. This is the gap, forced open: the
    write returns, the bytes do not arrive."""
    _create(skill)

    def _swallow(path, data, *, prefix=".tmp-"):
        return None                       # returns cleanly, writes nothing

    monkeypatch.setattr(store, "_atomic_write", _swallow)

    out = str(run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "width:44px;height:44px",
        "new_string": "width:76px;height:76px",
    }, CTX)))

    assert out.startswith("ERROR:")
    assert "did NOT take effect" in out
    assert "Do not tell the user it did" in out


def test_an_edit_says_the_user_cannot_see_it_yet(skill, apps_dir):
    """A write is not a publish. "Done" means present_app came back clean."""
    _create(skill)
    out = str(run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "width:44px;height:44px",
        "new_string": "width:76px;height:76px",
    }, CTX)))
    assert "present_app" in out
    assert "NOT" in out


def test_a_deletion_is_confirmed_by_what_is_gone(skill, apps_dir):
    """An empty new_string has no text to find, so the read-back proves the
    other half — otherwise every deletion would be refused as "did not land"."""
    _create(skill)
    run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "let running = false;",
        "new_string": "let running = false; // marker",
    }, CTX))
    out = str(run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": " // marker", "new_string": "",
    }, CTX)))
    assert not out.startswith("ERROR:")
    assert "the old text is gone from the file" in out
    assert "// marker" not in store.read_app("snake")


@pytest.mark.skipif(not __import__("shutil").which("node"), reason="node absent")
def test_an_edit_that_breaks_the_script_is_reported(skill, apps_dir):
    """`create_app_file` has parsed its own JavaScript since round 18 and an
    edit parsed nothing, so an edit could kill the script and the next thing
    that looked was `present_app` — after the model had already answered."""
    _create(skill)
    out = str(run(skill.execute_tool("app_html__edit_app_file", {
        "slug": "snake", "old_string": "function turn(d){",
        "new_string": "function turn(d){ {{{",
    }, CTX)))
    assert "no longer parses" in out
    assert "Fix that before you say anything to the user" in out


# ═════════════════════════════════════════════════════════════════════
# 5. The guidance the model is actually given
# ═════════════════════════════════════════════════════════════════════
#
# Asserted against `get_system_prompt_section()`, not against the .md file: the
# document is only guidance if it reaches the model, and round 18 moved it out
# of a tool result and into the prompt precisely because the delivery could
# fail. A rule in a file nobody reads is a rule that does not exist.

@pytest.fixture()
def prompt(apps_dir):
    return AppHtmlSkill().get_system_prompt_section() or ""


@pytest.mark.parametrize("needle", [
    "44 × 44",            # the floor, with its three sources named
    "64 × 64",            # the game control, which is not a link in a list
    "bottom third",       # thumb reach
    "dvh",                # not vh — the bottom row is off screen in vh
    "env(safe-area-inset",
    "orientation:landscape",
    "tabular-nums",
    "touch-action:manipulation",
    ":active",            # a phone has no hover
])
def test_the_prompt_carries_the_sizing_rules(prompt, needle):
    assert needle in prompt, f"{needle!r} is not in the system prompt"


def test_the_prompt_says_which_control_a_change_is_about(prompt):
    """Round 19's third defect: "make the button bigger" on a game with a D-pad
    changed the PLAY buttons."""
    assert "D-pad" in prompt
    assert "CHANGE THEM ALL" in prompt
    assert "Never ask which button they meant." in prompt


def test_the_prompt_forbids_claiming_an_unpublished_change(prompt):
    assert "Never say it is done until it is" in prompt
    assert "present_app returned" in prompt


def test_the_checklist_can_be_ticked_against_the_screen(prompt):
    """Every one of these is invisible to every gate — a 32 px D-pad throws
    nothing and renders perfectly — so the checklist is the only check."""
    for item in ("≥ 44 × 44 CSS px", "bottom third", "viewport-fit=cover",
                 "only in portrait"):
        assert item in prompt, item


# ═════════════════════════════════════════════════════════════════════
# 6. The platform route the runner asks — two callers, one URL
# ═════════════════════════════════════════════════════════════════════
#
# `?token=` is a browser frame and gets the sandboxed DOCUMENT. An account
# Bearer is the Toup client and gets the HANDLE — slug, title, revision, body.
# Answering the second with 401 is half of why an edit never arrived: told
# that revision 2 existed, the runner dropped its cached body, came here for
# the new one, and was refused.

@pytest.fixture()
def platform(monkeypatch):
    """`serve_artifact` with the agent hop stubbed, everything else real."""
    from app.api import artifact_proxy
    from app.services import auth_service

    async def _agent(user_id, db):
        return ("http://agent.test", "agent-key")

    class _Resp:
        status_code = 200
        content = b"<!doctype html><html><body>wrapped</body></html>"
        headers = {"x-toup-artifact-revision": "7"}

        @staticmethod
        def json():
            return {"slug": "snake", "title": "Snake", "revision": 7,
                    "html": "<!doctype html><html><body>raw</body></html>"}

    class _Client:
        calls = []

        async def get(self, url, **_kw):
            _Client.calls.append(url)
            return _Resp()

    class _User:
        id = "user-1"
        is_active = True

    async def _user(db, uid):
        return _User() if uid == "user-1" else None

    from app.services import agent_http
    monkeypatch.setattr(artifact_proxy, "_get_agent", _agent)
    monkeypatch.setattr(agent_http, "get_agent_http_client", lambda: _Client())
    monkeypatch.setattr(auth_service, "get_user_by_id", _user)
    _Client.calls = []
    return _Client


def _request(headers: dict):
    from starlette.requests import Request
    raw = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    return Request({"type": "http", "method": "GET", "path": "/", "headers": raw})


def _serve(headers=None, token=None):
    from app.api.artifact_proxy import serve_artifact
    return run(serve_artifact(
        "snake", _request(headers or {}), token=token, db=None,
    ))


def test_a_frame_with_a_scoped_token_still_gets_the_document(platform):
    from app.services.auth_service import create_artifact_token

    resp = _serve(token=create_artifact_token("user-1", "snake"))

    assert resp.media_type == "text/html; charset=utf-8"
    assert b"wrapped" in resp.body
    assert platform.calls[-1].endswith("/artifacts/snake")


def test_a_client_with_the_account_bearer_gets_the_handle(platform):
    from app.services.auth_service import create_access_token

    resp = _serve({"authorization": f"Bearer {create_access_token('user-1')}"})

    assert resp.media_type == "application/json"
    assert json.loads(resp.body)["revision"] == 7
    # The SOURCE route — the unwrapped bytes — not the document route.
    assert platform.calls[-1].endswith("/artifacts/snake/source")


def test_the_handle_is_never_cached(platform):
    from app.services.auth_service import create_access_token

    resp = _serve({"authorization": f"Bearer {create_access_token('user-1')}"})
    assert resp.headers["cache-control"] == "no-store"


def test_no_credential_at_all_is_still_refused(platform):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _serve()
    assert exc.value.status_code == 401


def test_a_cookie_is_not_a_credential_here(platform):
    """The module's rule, extended to the new branch. A cookie is attached by
    the browser to any request any page can cause, so honouring one would let
    an attacker's `<iframe src=".../artifacts/snake">` reach a victim's app on
    their ambient session. A Bearer header cannot be attached that way."""
    from fastapi import HTTPException
    from app.services.auth_service import create_access_token

    tok = create_access_token("user-1")
    with pytest.raises(HTTPException) as exc:
        _serve({"cookie": f"toup_sso={tok}"})
    assert exc.value.status_code == 401


def test_an_artifact_token_does_not_become_an_account_bearer(platform):
    """Credential separation: the scoped token fetches one static file and
    must not be upgradeable into the account's view of it."""
    from fastapi import HTTPException
    from app.services.auth_service import create_artifact_token

    with pytest.raises(HTTPException) as exc:
        _serve({"authorization":
                f"Bearer {create_artifact_token('user-1', 'snake')}"})
    assert exc.value.status_code == 401


def test_the_token_branch_wins_when_both_are_sent(platform):
    """The narrower credential decides, so a page that also carries a header
    cannot promote itself out of the document branch."""
    from app.services.auth_service import create_access_token, create_artifact_token

    resp = _serve({"authorization": f"Bearer {create_access_token('user-1')}"},
                  token=create_artifact_token("user-1", "snake"))

    assert resp.media_type == "text/html; charset=utf-8"
    assert platform.calls[-1].endswith("/artifacts/snake")


def test_a_disabled_account_gets_nothing(platform, monkeypatch):
    from fastapi import HTTPException
    from app.services import auth_service
    from app.services.auth_service import create_access_token

    class _Disabled:
        id = "user-1"
        is_active = False

    async def _user(db, uid):
        return _Disabled()

    monkeypatch.setattr(auth_service, "get_user_by_id", _user)
    with pytest.raises(HTTPException) as exc:
        _serve({"authorization": f"Bearer {create_access_token('user-1')}"})
    assert exc.value.status_code == 401


def test_the_guidance_still_reaches_the_prompt_without_its_frontmatter(prompt):
    """The document is indexed as a skill FILE and delivered as prose; the
    YAML header meant nothing in a prompt and was being shown to users."""
    assert "name: toup-frontend-design" not in prompt
    assert "Toup frontend design" in prompt
    assert "Read this **before** you write any UI" not in prompt
