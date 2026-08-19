"""The file library's pure rules (app/services/library_service.py) — no DB.

Allow-list of physical roots, deny rules inside them, display names,
name validation, storage-key containment. The fixture workspace is a
faithful copy of what the fleet survey found on real tenants on
2026-08-19 (founder tenant + pool-28): UUID dir, apps/ + vibecoding/ build
trees, dotdirs, image-tool workspace copies, harness scopes, stub PDFs,
LibreOffice preview caches, the boilerplate README.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.services import library_service as lib

USER = "871bac24-c366-42b5-b224-8802c73aef3a"
OTHER = "e5ec1759-84ac-45bb-b2ec-bc85fa932211"


def _w(path, data=b"x" * 2048):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


@pytest.fixture
def ws(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "agent_workspace_dir", str(tmp_path))
    root = str(tmp_path)
    # ── internals that must NEVER show ─────────────────────────────
    _w(f"{root}/.dashboard/state.json")
    _w(f"{root}/.whatsapp_auth/creds.json")
    _w(f"{root}/.cache/x")
    _w(f"{root}/.toup_bg_promoted", b"1")
    _w(f"{root}/apps/Nokia-Snake-Arcade/App.tsx")
    _w(f"{root}/apps/Nokia-Snake-Arcade/package.json")
    _w(f"{root}/apps/Nokia-Snake-Arcade/node_modules/left-pad/README.md")
    _w(f"{root}/vibecoding/todo/index.html")
    _w(f"{root}/toup-code/{USER}/proj/main.py")
    _w(f"{root}/{USER}/README.md", b"# Toup Agent Workspace\n\nCreated: ...\n")
    _w(f"{root}/{USER}/scratch.py")            # code at the user root: not a deliverable
    _w(f"{root}/notes.json")                    # data dump at the ws root: not a deliverable
    # harness scopes + stubs
    _w(f"{root}/generated/e2e-final/c2abccc5c7254d6eb480fcac37572070_A_fox_safe.png", b"F" * 5000)
    _w(f"{root}/generated/edit-e2e/3f0f6ddcea254e8a9df2020fa6365e32_edited_A.png")
    _w(f"{root}/generated/roundtrip-user/2e6f3c9ededb48d181099f4b6d2f8501_inbound_test.png")
    _w(f"{root}/generated/{OTHER}/0000aaaa0000aaaa0000aaaa0000aaaa_other_users_file.pdf", b"P" * 4000)
    _w(f"{root}/generated/shared/5535d3bc3a4a4d80ae71745c99b5f85c_x.pdf", b"%PDF" + b"0" * 948)  # 952 B stub
    _w(f"{root}/generated/shared/299641f3873042e7bff2f8f454ea46df_foo.pdf", b"%PDF" + b"0" * 1592)
    _w(f"{root}/generated/shared/6a4c975297d04739a694ef30f131b8f0_test.pdf", b"%PDF" + b"0" * 1594)
    _w(f"{root}/generated/{USER}/b271948939724bb7a37d2855bde3e2c7_e2e_test.png")
    _w(f"{root}/generated/{USER}/c9a145bfa1934b609c4b9250708b91c8_hq_test.png")
    _w(f"{root}/generated/{USER}/deadbeefdeadbeefdeadbeefdeadbeef_empty.pdf", b"")
    # root copies (image tools drop one next to the attachment)
    _w(f"{root}/A_fox_safe.png", b"F" * 5000)                 # copy of a DENIED scope file
    _w(f"{root}/muscular-veiny-hand-steering-wheel.png", b"M" * 3000)  # copy of a real attachment
    _w(f"{root}/e2e_test.png")
    _w(f"{root}/hq_test.png")
    _w(f"{root}/edited_A.png")
    _w(f"{root}/src_apple.png")
    # ── the real library ───────────────────────────────────────────
    _w(f"{root}/generated/{USER}/03291e2d652b4e20a5af0b3075ca3d71_muscular-veiny-hand-steering-wheel.png", b"M" * 3000)
    _w(f"{root}/generated/{USER}/045a5029509d462293e94689f200986a_IMG_3145.jpg", b"J" * 7000)
    _w(f"{root}/generated/{USER}/7d4d525217c1464aa3803883982c9ae9_Resume_2.pdf", b"%PDF" + b"R" * 35000)
    _w(f"{root}/generated/shared/f112c3d24b3246798cfffed224f64bc8_uoft-events.docx", b"D" * 40000)
    _w(f"{root}/generated/shared/f112c3d24b3246798cfffed224f64bc8_uoft-events.docx.preview.pdf", b"%PDF" * 5000)
    _w(f"{root}/generated/drafts/report.pdf", b"%PDF" + b"W" * 3000)      # write_file redirect, no user scope
    _w(f"{root}/generated/summary.md", b"# summary\n" * 20)                # exec sweep placement
    _w(f"{root}/{USER}/generated/plan.md", b"# plan\n" * 20)               # write_file redirect, user scope
    _w(f"{root}/{USER}/generated/q3/budget.xlsx", b"X" * 9000)
    _w(f"{root}/{USER}/thoughts.md", b"# thoughts\n" * 10)                 # model-written doc at user root
    _w(f"{root}/project-management-tools-comparison-2026.md", b"# pm\n" * 400)  # legacy root doc
    return root


# ── Allow-list ────────────────────────────────────────────────────────

def _live_rels(user):
    out = {}
    for c in lib.iter_physical_candidates(user):
        tag, rel = lib.split_key(c.key)
        base = rel.rsplit("/", 1)[-1]
        if c.denied or lib.is_junk(base, c.size, path_for_sniff=c.path if base.lower() == "readme.md" else None):
            continue
        out[c.key] = c
    return out


def test_allowlist_yields_only_deliverable_roots(ws):
    keys = set(_live_rels(USER))
    expected = {
        f"gen:{USER}/03291e2d652b4e20a5af0b3075ca3d71_muscular-veiny-hand-steering-wheel.png",
        f"gen:{USER}/045a5029509d462293e94689f200986a_IMG_3145.jpg",
        f"gen:{USER}/7d4d525217c1464aa3803883982c9ae9_Resume_2.pdf",
        "gen:shared/f112c3d24b3246798cfffed224f64bc8_uoft-events.docx",
        "gen:drafts/report.pdf",
        "gen:summary.md",
        "uws:generated/plan.md",
        "uws:generated/q3/budget.xlsx",
        "uws:thoughts.md",
        "ws:project-management-tools-comparison-2026.md",
        # root copies are candidates here; the SYNC dedupes them against the
        # attachment they duplicate (test_library_api covers that).
        "ws:muscular-veiny-hand-steering-wheel.png",
        "ws:A_fox_safe.png",
    }
    assert keys == expected, sorted(keys ^ expected)


def test_denied_dirs_and_dotdirs_are_never_walked(ws):
    joined = " ".join(c.key for c in lib.iter_physical_candidates(USER))
    for forbidden in ("apps/", "vibecoding/", "toup-code/", ".dashboard", ".whatsapp_auth", ".cache",
                      "node_modules", ".toup_bg_promoted", "scratch.py", "notes.json"):
        assert forbidden not in joined, forbidden


def test_foreign_scope_and_harness_scopes_are_denied_not_imported(ws):
    denied = {c.key for c in lib.iter_physical_candidates(USER) if c.denied}
    assert f"gen:{OTHER}/0000aaaa0000aaaa0000aaaa0000aaaa_other_users_file.pdf" in denied
    assert "gen:e2e-final/c2abccc5c7254d6eb480fcac37572070_A_fox_safe.png" in denied
    assert "gen:edit-e2e/3f0f6ddcea254e8a9df2020fa6365e32_edited_A.png" in denied
    live = _live_rels(USER)
    assert not any(OTHER in k or "e2e-final" in k or "edit-e2e" in k or "roundtrip" in k for k in live)


def test_the_bare_uuid_dir_and_boilerplate_readme_are_hidden(ws):
    live = _live_rels(USER)
    assert "uws:README.md" not in live
    assert not any(k.startswith(f"ws:{USER}") for k in live)


# ── Deny rules inside allowed roots ──────────────────────────────────

@pytest.mark.parametrize("name,size", [
    ("e2e_test.png", 2048), ("hq_test.png", 2048), ("edited_A.png", 2048), ("edited_B.png", 2048),
    ("src_apple.png", 2048), ("inbound_test.png", 2048), ("user_upload.png", 2048),
    ("x.pdf", 952), ("foo.pdf", 1596), ("test.pdf", 1598), ("tmp.txt", 10), ("sample3.docx", 500),
    ("report.docx.preview.pdf", 90000), ("empty.pdf", 0), (".hidden.md", 100),
    ("5535d3bc3a4a4d80ae71745c99b5f85c.pdf", 3000),                   # bare id
    ("871bac24-c366-42b5-b224-8802c73aef3a.png", 3000),               # bare uuid
    ("6a4c975297d04739a694ef30f131b8f0_x.pdf", 952),                  # stub through its storage name
    ("anything.pdf", 999),                                            # empty-reportlab stub
])
def test_junk_names_are_junk(name, size):
    assert lib.is_junk(name, size) is True


@pytest.mark.parametrize("name,size", [
    ("Q3 test results.pdf", 40000), ("IMG_3145.jpg", 3221950), ("resume.pdf", 35425),
    ("uoft-events.docx", 40604), ("summary.md", 200), ("testimonials.docx", 5000),
    ("045a5029509d462293e94689f200986a_IMG_3145.jpg", 3221950), ("x-ray-results.pdf", 5000),
    ("latest.pdf", 1001), ("notes (2).md", 300),
])
def test_real_names_survive(name, size):
    assert lib.is_junk(name, size) is False


def test_boilerplate_readme_is_sniffed_not_name_matched(tmp_path):
    p = tmp_path / "README.md"
    p.write_bytes(b"# Toup Agent Workspace\n\nCreated: 2026-05-11")
    assert lib.is_junk("README.md", p.stat().st_size, path_for_sniff=str(p)) is True
    p.write_bytes(b"# My project README\n\nreal content")
    assert lib.is_junk("README.md", p.stat().st_size, path_for_sniff=str(p)) is False


# ── Names ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("stored,shown", [
    ("045a5029509d462293e94689f200986a_IMG_3145.jpg", "IMG_3145.jpg"),
    ("871bac24-c366-42b5-b224-8802c73aef3a_report.pdf", "report.pdf"),
    ("report.pdf", "report.pdf"),
    ("shared/f112c3d24b3246798cfffed224f64bc8_uoft.docx", "uoft.docx"),
    ("045a5029509d462293e94689f200986a_", "045a5029509d462293e94689f200986a_"),  # nothing after → unchanged
    ("abc_def.png", "abc_def.png"),                                             # not a 32-hex prefix
])
def test_display_name_strips_storage_prefix(stored, shown):
    assert lib.display_name(stored) == shown


@pytest.mark.parametrize("bad", ["", "   ", ".", "..", ".env", "a/b", "a\\b", "x\x00y", "a\nb", "z" * 201])
def test_validate_name_rejects(bad):
    with pytest.raises(lib.InvalidName):
        lib.validate_name(bad)


def test_validate_name_accepts_ordinary_names():
    assert lib.validate_name("  Q3 report (final).pdf ") == "Q3 report (final).pdf"
    assert lib.validate_name("Rapport été.docx") == "Rapport été.docx"


def test_safe_storage_filename_keeps_extension_and_drops_paths():
    assert lib.safe_storage_filename("../../etc/passwd") == "passwd"
    assert lib.safe_storage_filename("weird\x00name?.PDF") == "weirdname_.PDF"
    assert lib.safe_storage_filename("") == "file"
    long = "a" * 300 + ".docx"
    out = lib.safe_storage_filename(long)
    assert out.endswith(".docx") and len(out) <= 150


# ── Storage-key containment ──────────────────────────────────────────

def test_physical_path_resolves_inside_its_root(ws):
    p = lib.physical_path(USER, f"gen:{USER}/7d4d525217c1464aa3803883982c9ae9_Resume_2.pdf")
    assert p == os.path.realpath(f"{ws}/generated/{USER}/7d4d525217c1464aa3803883982c9ae9_Resume_2.pdf")
    assert lib.physical_path(USER, "uws:thoughts.md") == os.path.realpath(f"{ws}/{USER}/thoughts.md")
    assert lib.physical_path(USER, "ws:project-management-tools-comparison-2026.md").startswith(os.path.realpath(ws))


@pytest.mark.parametrize("key", [
    "gen:../apps/Nokia-Snake-Arcade/App.tsx", "gen:/etc/passwd", "uws:../.whatsapp_auth/creds.json",
    "ws:../../etc/passwd", "gen:a/../../..", "bogus:x", "gen:", "", "gen", "ws:..",
])
def test_physical_path_refuses_escapes(ws, key):
    with pytest.raises(ValueError):
        lib.physical_path(USER, key)


def test_physical_path_refuses_symlink_escape(ws):
    os.symlink("/etc", f"{ws}/generated/{USER}/etc_link")
    with pytest.raises(ValueError):
        lib.physical_path(USER, f"gen:{USER}/etc_link/passwd")


# ── Classification / metadata helpers ────────────────────────────────

@pytest.mark.parametrize("name,kind", [
    ("a.pdf", "document"), ("a.MD", "document"), ("a.xlsx", "document"), ("a.png", "image"),
    ("a.HEIC", "image"), ("a.mp3", "audio"), ("a.mov", "video"), ("a.zip", "archive"), ("a.bin", "other"),
])
def test_kind_of(name, kind):
    assert lib.kind_of(name) == kind


def test_kind_falls_back_to_mime_family():
    assert lib.kind_of("blob", "image/png") == "image"
    assert lib.kind_of("blob", "text/plain") == "document"


def test_guess_mime_prefers_extension_overrides_then_declared():
    assert lib.guess_mime("notes.md") == "text/markdown"
    assert lib.guess_mime("x.docx", "application/octet-stream") == \
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert lib.guess_mime("weird.bin", "application/x-foo") == "application/x-foo"
    assert lib.guess_mime("weird.bin") == "application/octet-stream"


def test_human_size():
    assert lib.human_size(0) == "0 B"
    assert lib.human_size(1023) == "1023 B"
    assert lib.human_size(35425) == "34.6 KB"
    assert lib.human_size(2486315) == "2.4 MB"
    assert lib.human_size(3 * 1024 ** 3) == "3.00 GB"


def test_split_virtual_path_rejects_traversal():
    assert lib.split_virtual_path("") == []
    assert lib.split_virtual_path("/Documents//Q3 report.pdf/") == ["Documents", "Q3 report.pdf"]
    for bad in ("../x", "Documents/../..", "./x", "a/./b", "a\x00"):
        with pytest.raises(ValueError):
            lib.split_virtual_path(bad)


def test_default_placement():
    assert lib.default_system_key("upload", "image") == "uploads"
    assert lib.default_system_key("agent", "image") == "images"
    assert lib.default_system_key("agent", "document") == "documents"
    assert lib.default_system_key("agent", "archive") == "documents"


# ── Content-aware rules (learned from the 2026-08-19 fleet dry run) ─────

def _docx(path, text: str):
    import zipfile
    body = "".join(f"<w:p><w:r><w:t>{t}</w:t></w:r></w:p>" for t in text.split("\n") if t)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/styles.xml", "<w:styles>" + "x" * 30000 + "</w:styles>")  # the 36 KB skeleton
        z.writestr("word/document.xml", f"<w:document><w:body>{body}</w:body></w:document>")
    return path


def test_office_stub_is_junk_regardless_of_name_and_size(tmp_path):
    empty = _docx(str(tmp_path / "Quarterly plan.docx"), "")
    real = _docx(str(tmp_path / "x.docx"), "Q3 revenue grew 14% on the back of the new pricing tiers.\nMore.")
    assert os.path.getsize(empty) > 30000            # size cannot tell them apart
    assert lib.office_is_empty(empty, "docx") is True
    assert lib.office_is_empty(real, "docx") is False
    assert lib.is_junk("Quarterly plan.docx", os.path.getsize(empty), path_for_sniff=empty) is True
    # a placeholder NAME on a document with real content is a lazy name, not junk
    assert lib.is_junk("x.docx", os.path.getsize(real), path_for_sniff=real) is False
    assert lib.is_junk("test.docx", 36599, path_for_sniff=real) is False
    # unreadable → never hidden on that basis
    assert lib.office_is_empty(str(tmp_path / "missing.docx"), "docx") is None
    bogus = tmp_path / "bogus.docx"; bogus.write_bytes(b"D" * 40000)
    assert lib.is_junk("bogus.docx", 40000, path_for_sniff=str(bogus)) is False


def test_placeholder_names_are_junk_only_when_small():
    assert lib.is_junk("test.pdf", 1598) is True         # docgen smoke test
    assert lib.is_junk("foo.md", 40) is True
    assert lib.is_junk("x.png", 7000) is True
    assert lib.is_junk("test.pdf", 40000) is False        # a real 40 KB document named lazily
    assert lib.is_junk("output.xlsx", 20000) is False
    assert lib.is_junk("e2e_test.png", 2307675) is True   # incident names stay unconditional
    assert lib.is_junk("hq_test.png", 1740738) is True


def test_pptx_and_xlsx_emptiness(tmp_path):
    import zipfile
    p = str(tmp_path / "deck.pptx")
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("ppt/slides/slide1.xml", "<p:sld><a:t></a:t></p:sld>")
    assert lib.office_is_empty(p, "pptx") is True
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("ppt/slides/slide1.xml", "<p:sld><a:t>Welcome to the quarterly business review</a:t></p:sld>")
    assert lib.office_is_empty(p, "pptx") is False
    x = str(tmp_path / "book.xlsx")
    with zipfile.ZipFile(x, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData/></worksheet>")
    assert lib.office_is_empty(x, "xlsx") is True
    with zipfile.ZipFile(x, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", "<worksheet><sheetData><row><c><v>42</v></c></row></sheetData></worksheet>")
    assert lib.office_is_empty(x, "xlsx") is False
