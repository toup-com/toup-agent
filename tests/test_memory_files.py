"""Memory files v3 — the canon module (docs/memory/rebuild-2026-08-v3.md §1).

Pure, stdlib-only, no DB: sections and slugs, the description regex, the
bullet lint that keeps the one voice, body parse/render round trips, the
truncation rule, and the SHARED injection renderer both assemblers use.

Round 8's version of this file pinned a category→section map. v3 has no
such map in canon — categories are not the router any more — so what is
pinned here is the slug namespace, which is.
"""

import pytest

from app.memory_files import (
    normalize_bullet,
    normalize_description,
    ALWAYS_INJECTED_SLUGS,
    CAP_PROFILE,
    CURRENT_CONTEXT_LAYERS,
    CURRENT_CONTEXT_SLUG,
    DESCRIPTION_RE,
    HEADING_CURRENT_CONTEXT,
    HEADING_INDEX,
    HEADING_LEARNED,
    HEADING_PROFILE,
    LEARNED_SLUG,
    MAX_BODY_CHARS,
    PROFILE_SLUG,
    SECTION_LABEL,
    SECTION_ORDER,
    SYSTEM_FILES,
    TRUNCATION_NOTE,
    FileSection,
    area_slug,
    body_is_empty,
    bullet_problem,
    description_problem,
    extract_links,
    index_line,
    is_valid_slug,
    parse_bullets,
    person_slug,
    render_bullets,
    render_user_brain,
    section_of_slug,
    slugify,
    title_from_slug,
    topic_slug,
    truncate_body,
)


# ── Sections and system files ─────────────────────────────────────────

def test_sections_are_the_five_v3_sections_in_display_order():
    assert [s.value for s in SECTION_ORDER] == [
        "you", "people", "topics", "areas", "learned"
    ]
    for section in SECTION_ORDER:
        assert SECTION_LABEL[section]
    # Round 8's catch-alls are gone from canon — no Preferences, no
    # Knowledge, no Working. A file with nowhere to go is a routing bug,
    # not a drawer to open.
    assert not {"preferences", "knowledge", "working", "profile"} & {
        s.value for s in SECTION_ORDER
    }


def test_the_three_system_files_are_the_always_injected_ones():
    assert set(SYSTEM_FILES) == {PROFILE_SLUG, CURRENT_CONTEXT_SLUG, LEARNED_SLUG}
    assert set(ALWAYS_INJECTED_SLUGS) == set(SYSTEM_FILES)
    for slug, spec in SYSTEM_FILES.items():
        assert is_valid_slug(slug), slug          # they ride URLs
        assert FileSection(spec["section"]) in SECTION_ORDER
        # A system file's description is the worked example the writer
        # copies, so it must satisfy the same regex it is held to.
        assert description_problem(spec["description"]) is None, slug


def test_section_of_slug_reads_the_namespace_and_refuses_to_guess():
    assert section_of_slug(PROFILE_SLUG) == FileSection.YOU
    assert section_of_slug(LEARNED_SLUG) == FileSection.LEARNED
    assert section_of_slug("people/majid-tajik") == FileSection.PEOPLE
    assert section_of_slug("topics/music") == FileSection.TOPICS
    assert section_of_slug("areas/toup") == FileSection.AREAS
    # A round-8 slug names no v3 namespace. It must answer None rather than
    # falling back to a section — that fallback is how a `knowledge` row
    # would surface inside a v3 section it was never written for.
    for legacy in ("knowledge", "preferences", "working", "profile", "people"):
        assert section_of_slug(legacy) is None, legacy


# ── Slugs ─────────────────────────────────────────────────────────────

def test_slugs_are_safe_and_unicode_capable():
    assert slugify("Majid Tajik") == "majid-tajik"
    assert slugify("  IELTS — Prep!  ") == "ielts-prep"
    assert slugify("محمد السالم") == "محمد-السالم"  # Persian survives
    assert person_slug("Majid Tajik") == "people/majid-tajik"
    assert topic_slug("Music") == "topics/music"
    assert area_slug("Toup") == "areas/toup"
    assert person_slug("!!!") is None
    assert is_valid_slug("people/majid-tajik")
    assert not is_valid_slug("../../etc/passwd")
    assert not is_valid_slug("a/b/c")
    assert not is_valid_slug("has_underscore")
    assert not is_valid_slug("")


def test_title_from_slug_is_a_last_resort_only():
    assert title_from_slug(PROFILE_SLUG) == "Profile"
    assert title_from_slug("people/majid-tajik") == "Majid tajik"


# ── Descriptions ──────────────────────────────────────────────────────

def test_description_pattern_is_enforced_not_suggested():
    good = (
        "Your IELTS preparation — tutor, dates and band targets; "
        "read when IELTS or the exam comes up."
    )
    assert DESCRIPTION_RE.match(good)
    assert description_problem(good) is None

    for bad, why in [
        ("", "empty"),
        ("Music", "no structure at all"),
        ("Music - things; read when music comes up.", "hyphen, not an em dash"),
        ("Music — things, read when music comes up.", "comma, not a semicolon"),
        ("Music — things; read when music comes up", "no trailing period"),
        ("Music — things; when music comes up.", "missing the 'read when' trigger"),
        ("Music — things; read when music comes up.\nand more", "two lines"),
    ]:
        assert description_problem(bad) is not None, why


def test_there_is_no_templated_description_helper():
    """Round 8's `default_purpose` minted "Nariman — someone in your life;
    read when they come up." for every person file, and the curation gate
    that was supposed to replace it refused files with fewer than two
    entries — so the mad-lib was permanent on exactly the files that got
    one. v3 §1.4 deletes it: a file is born with a real description or the
    create op is rejected."""
    import app.memory_files as canon

    assert not hasattr(canon, "default_purpose")


# ── Bullet voice ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bullet", [
    "uses an Android phone",
    "has a Claude Max 20x subscription; uses the Claude phone app and Claude Desktop",
    "IELTS exam booked for Aug 30, 2026",
    "goal: build visible abs / six pack, wants the fastest route",
    "teaches by sending an upgraded word; Nariman writes a sentence with it, then he corrects it",
    "می‌خواهد هر روز صبح خلاصه ایمیل‌ها را بگیرد",
    # Two words IS a fact. The three-word floor this file used to pin
    # rejected the design doc's own worked example.
    "likes Googoosh",
])
def test_the_house_voice_passes(bullet):
    assert bullet_problem(bullet) is None, bullet


@pytest.mark.parametrize("bullet,why", [
    ("You use an Android phone", "the subject is implied, never restated"),
    ("Your name is Nariman", "same"),
    ("The user prefers dark mode", "third person about 'the user'"),
    ("wants", "a single word is a fragment, not a fact"),
    ("", "empty"),
    ("x" * 401, "over 400 chars"),
    ("ran job 6f1c2b9a-1111-2222-3333-444455556666 yesterday", "a UUID"),
    ("gmail briefing uses max_results=1 every morning", "a tool parameter"),
    ("first line\nsecond line", "a bullet is one line"),
])
def test_the_lint_rejects_what_round_8_stored(bullet, why):
    assert bullet_problem(bullet) is not None, why


def test_a_long_hex_id_is_rejected_but_ordinary_numbers_are_not():
    assert bullet_problem("session id is 0123456789abcdef0123") is not None
    # A phone number, a year, a price: none of these are ids.
    assert bullet_problem("pays 98.90 CAD a month for the Max plan") is None
    assert bullet_problem("moved to Toronto in 2019 for the UofT program") is None


# ── Bodies ────────────────────────────────────────────────────────────

def test_bullets_round_trip_and_ignore_non_bullet_lines():
    body = (
        "## Today\nSome prose about today.\n\n"
        "- uses an Android phone\n"
        "* likes Googoosh\n"
        "-   spaced oddly\n"
    )
    assert parse_bullets(body) == [
        "uses an Android phone", "likes Googoosh", "spaced oddly",
    ]
    assert render_bullets(["a b c", "  ", "d e f"]) == "- a b c\n- d e f"
    assert render_bullets([]) == ""
    assert body_is_empty("") and body_is_empty(None) and body_is_empty("   ")


def test_links_are_extracted_in_first_seen_order_without_duplicates():
    body = "- taught by [[people/majid]]\n- also [[people/majid]] and [[areas/ielts]]"
    assert extract_links(body) == ["people/majid", "areas/ielts"]
    assert extract_links("") == []


def test_truncation_lands_between_bullets_and_says_so():
    """Half a bullet is a FALSE fact, not a short one: "allergic to" reads
    as a complete predicate."""
    body = "\n".join(f"- fact number {i} about this person" for i in range(40))
    out = truncate_body(body, 200)
    assert len(out) <= 200
    assert out.endswith(TRUNCATION_NOTE)
    for line in out.splitlines():
        assert line == TRUNCATION_NOTE or line.startswith("- ")
    # Under the cap, nothing changes at all.
    assert truncate_body("- short", 500) == "- short"


# ── The shared injection renderer ─────────────────────────────────────

def test_render_user_brain_has_the_contract_shape():
    out = render_user_brain(
        profile_body="- uses an Android phone",
        current_context_body="## Today\nPreparing for the IELTS exam.",
        learned_body="- answer in Farsi when asked in Farsi",
        index=[("IELTS", "Your IELTS prep — dates; read when IELTS comes up."),
               ("Music", None)],
        relevant=[("IELTS", "- exam booked for Aug 30, 2026")],
    )
    # No heading of its own: the caller owns `# User Brain`, because the
    # injection fence binds to that exact literal.
    assert not out.startswith("# User Brain")
    order = [out.index(h) for h in (
        HEADING_PROFILE, HEADING_CURRENT_CONTEXT, HEADING_LEARNED, HEADING_INDEX,
    )]
    assert order == sorted(order)
    assert "- IELTS — Your IELTS prep — dates; read when IELTS comes up." in out
    assert "- Music" in out                      # a description-less file still lists
    assert out.rindex("## IELTS") > out.index(HEADING_INDEX)  # whole file last


def test_render_user_brain_omits_what_is_empty():
    assert render_user_brain() == ""
    only_profile = render_user_brain(profile_body="- uses an Android phone")
    assert only_profile == "## Profile\n- uses an Android phone"


def test_render_user_brain_enforces_every_cap():
    big = "\n".join(f"- fact number {i} about this person" for i in range(400))
    out = render_user_brain(
        profile_body=big, current_context_body=big, learned_body=big,
        index=[(f"File {i}", f"F{i} — s; read when x.") for i in range(100)],
        relevant=[("Alpha", big), ("Beta", big), ("Gamma", big)],
    )
    assert out.count(TRUNCATION_NOTE) >= 3
    assert out.count("\n- File ") <= 40           # MAX_INDEX_LINES
    assert "## Alpha" in out and "## Beta" in out
    assert "## Gamma" not in out                  # MAX_RELEVANT_FILES = 2
    profile_block = out.split(HEADING_CURRENT_CONTEXT)[0]
    assert len(profile_block) <= CAP_PROFILE + len(HEADING_PROFILE) + 2


def test_body_cap_is_a_real_number_and_the_layers_are_named():
    assert MAX_BODY_CHARS == 8 * 1024
    assert CURRENT_CONTEXT_LAYERS[0] == "Today"
    assert CURRENT_CONTEXT_LAYERS[-1] == "Past 12 months"
    assert index_line("Music", None) == "- Music"


# ── Legacy half ───────────────────────────────────────────────────────

def test_the_round_8_map_survives_only_behind_the_legacy_prefix():
    """WS-2 deletes it with `memory_file_service`; WS-5's migration reads it
    to interpret round-8 file assignments. Nothing v3 may touch it, and the
    naming is what makes that checkable."""
    import app.memory_files as canon

    assert canon.legacy_default_slug_for("goals", "user") == "areas/work"
    assert canon.legacy_default_slug_for("corrections", "agent") == "learned"
    assert canon.legacy_section_of_slug("knowledge") == canon.LegacyFileSection.KNOWLEDGE
    # The v3 names must not resolve to round-8 values.
    assert not hasattr(canon, "USER_CATEGORY_SECTION")
    assert not hasattr(canon, "default_slug_for")
    assert not hasattr(canon, "section_for")


# ── One sentence takes no full stop ───────────────────────────────────
#
# The contract has always said so and nothing enforced it, so the rule was a
# preference the writer kept about half the time. The founder's migrated
# Profile came back as four sentences each ending in a period while the same
# writer's turn path produces "listens to Googoosh and Ebi constantly" — one
# corpus in two voices, which is what "one consistent voice" forbids.


def test_a_single_sentence_loses_its_full_stop():
    assert normalize_bullet("uses an Android phone.") == "uses an Android phone"


def test_a_bullet_without_one_is_untouched():
    assert normalize_bullet("uses an Android phone") == "uses an Android phone"


def test_two_sentences_keep_theirs():
    """The rule is "unless the bullet holds more than one sentence"."""
    text = "exam booked for Aug 30, 2026. Targeting band 7.5 overall."
    assert normalize_bullet(text) == text


def test_a_semicolon_join_is_still_one_sentence():
    assert normalize_bullet(
        "runs every morning at 7; eats breakfast afterwards."
    ) == "runs every morning at 7; eats breakfast afterwards"


def test_an_ellipsis_is_left_alone():
    """A trailing "..." is not a full stop and truncation must stay visible."""
    assert normalize_bullet("wants to...") == "wants to..."


def test_persian_is_unaffected():
    fa = "هر روز صبح ساعت ۷ می‌دود و بعدش صبحانه می‌خورد"
    assert normalize_bullet(fa) == fa


# ── Description punctuation is repaired, not fatal ────────────────────
#
# CI 32436489185: the writer tried TWICE in one batch to open the person file
# P06 needs — `people/majid-tajik`, then `people/majid` — and both died on the
# description punctuation, taking every link that pointed at them with them.
# The routing was right; a full stop was missing. Third time in this rebuild
# that a formatting refusal cost the substance.


def test_a_missing_full_stop_is_added():
    assert normalize_description(
        "Their IELTS tutor — how they teach; read when IELTS comes up"
    ) == "Their IELTS tutor — how they teach; read when IELTS comes up."


def test_a_plain_hyphen_becomes_the_em_dash():
    assert normalize_description(
        "Their IELTS tutor - how they teach; read when IELTS comes up."
    ) == "Their IELTS tutor — how they teach; read when IELTS comes up."


def test_a_comma_before_read_when_becomes_a_semicolon():
    assert normalize_description(
        "Their IELTS tutor — how they teach, read when IELTS comes up."
    ) == "Their IELTS tutor — how they teach; read when IELTS comes up."


def test_all_three_at_once():
    assert description_problem(normalize_description(
        "Their IELTS tutor - how they teach, read when IELTS comes up"
    )) is None


def test_a_conforming_description_is_untouched():
    good = "Their IELTS tutor — how they teach; read when IELTS comes up."
    assert normalize_description(good) == good


def test_only_the_FIRST_dash_is_promoted():
    """Later dashes are prose and must survive."""
    out = normalize_description(
        "Their tutor - a part-time teacher - who they are; read when it comes up"
    )
    assert out.count("—") == 1
    assert "part-time" in out


def test_something_that_is_not_a_description_is_still_refused():
    """The repair is punctuation only. It cannot invent the shape."""
    assert description_problem(normalize_description("a tutor")) is not None


def test_an_em_dash_without_spaces_is_spaced():
    assert normalize_description(
        "Their IELTS tutor—how they teach; read when IELTS comes up."
    ) == "Their IELTS tutor — how they teach; read when IELTS comes up."


def test_a_capitalised_read_when_is_lowered():
    assert description_problem(normalize_description(
        "Their IELTS tutor — how they teach; Read when IELTS comes up."
    )) is None


def test_the_complaint_shows_what_was_actually_written():
    """Twice in this rebuild the fix for this complaint had to be guessed at,
    because the message named the shape it wanted and never the string it
    got."""
    problem = description_problem("a tutor")
    assert problem and "'a tutor'" in problem, problem
