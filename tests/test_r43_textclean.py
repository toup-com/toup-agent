"""One decode, correctly — the em dash the founder's brief mangled.

Three defects met on one string (see `app/connectors/textclean`). Each
of them is silent: the pipeline runs, the brief posts, and the subject
is simply wrong. The founder's own screenshot is the fixture.
"""

from app.agent.automations.ledger import strip_html
from app.connectors.textclean import (
    clean_provider_text,
    decode_mail_header,
    repair_mojibake,
    unescape_once,
)


EM_DASH = "R29-D live loop test — Gmail push"


def _wrong(text: str, times: int = 1, enc: str = "latin-1") -> str:
    """UTF-8 bytes read as `enc`, `times` times — how mojibake is made."""
    for _ in range(times):
        text = text.encode("utf-8").decode(enc)
    return text


class TestRfc2047:
    def test_an_encoded_word_subject_decodes(self):
        assert decode_mail_header(
            "=?UTF-8?Q?R29-D_live_loop_test_=E2=80=94_Gmail_push?="
        ) == EM_DASH

    def test_base64_encoded_words_decode(self):
        assert decode_mail_header("=?utf-8?B?U2VjdXJpdHkgYWxlcnQ=?=") == "Security alert"

    def test_a_plain_subject_is_untouched(self):
        assert decode_mail_header(EM_DASH) == EM_DASH

    def test_a_malformed_encoded_word_survives_as_itself(self):
        # A header we cannot read is still a header; showing its raw form
        # beats showing nothing.
        raw = "=?NOSUCHCHARSET?Q?hi?="
        assert decode_mail_header(raw) == raw

    def test_empty_is_empty(self):
        assert decode_mail_header(None) == ""


class TestMojibakeRepair:
    def test_the_founders_double_wrong_decode(self):
        # The subject was mangled twice — once by the old outbound header
        # path, once on the way back in — so a single round trip is not
        # enough and a one-pass repair would have called it done.
        broken = _wrong(EM_DASH, 2)
        assert broken != EM_DASH
        assert repair_mojibake(broken) == EM_DASH

    def test_a_single_wrong_decode(self):
        assert repair_mojibake(_wrong(EM_DASH, 1)) == EM_DASH

    def test_the_cp1252_flavour(self):
        assert repair_mojibake(_wrong(EM_DASH, 1, "cp1252")) == EM_DASH

    def test_correct_text_is_never_rewritten(self):
        for good in (
            EM_DASH,
            "Ça va? déjà vu, naïve",
            "ação e coração",
            "Grüße aus München",
            "سلام دنیا",
            "ship it \U0001f680 — now",
            "Total: €30 · £20 · ¥100",
        ):
            assert repair_mojibake(good) == good


class TestUnescapeOnce:
    def test_a_literal_ampersand_entity_survives_the_layers(self):
        # The bug: three layers each unescaped, so a sender who wrote a
        # literal "&amp;" had it turned into "&".
        once = strip_html("Tom &amp;amp; Jerry")
        assert once == "Tom &amp; Jerry"
        assert strip_html(once, unescape=False) == "Tom &amp; Jerry"
        assert strip_html(strip_html(once, unescape=False), unescape=False) == "Tom &amp; Jerry"

    def test_the_old_triple_was_lossy(self):
        # Kept as the proof of what changed: the previous shape reached "&".
        assert strip_html(strip_html(strip_html("Tom &amp;amp; Jerry"))) == "Tom & Jerry"

    def test_tags_still_go_at_every_layer(self):
        assert strip_html("<b>bold</b> text", unescape=False) == "bold text"

    def test_an_address_keeps_its_angle_brackets(self):
        raw = "Google <no-reply@accounts.google.com>"
        assert strip_html(raw) == raw
        assert strip_html(raw, unescape=False) == raw


class TestTheOneDoor:
    def test_header_path_decodes_then_unescapes_then_repairs(self):
        assert clean_provider_text(
            "=?UTF-8?Q?Tom_=26amp=3B_Jerry?=", header=True
        ) == "Tom & Jerry"

    def test_body_path_does_not_decode_encoded_words(self):
        # A message quoting an encoded word is quoting it.
        quoted = "he wrote =?utf-8?q?hi?= in the body"
        assert clean_provider_text(quoted) == quoted

    def test_unescape_once_is_the_only_unescape(self):
        assert unescape_once("&amp;amp;") == "&amp;"
