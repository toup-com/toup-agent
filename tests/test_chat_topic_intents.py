"""Navigation tasks follow user intent, with conservative, owned-row aliases."""

from datetime import datetime
import json
import uuid

import pytest

from app.services import chat_topics as topics


def row(id, content, role="user", conversation="chat", **extra):
    return {"id": id, "content": content, "role": role,
            "conversation_id": conversation, "channel": "web", **extra}


QUALITY_CASES = {
    "pause_control_test": [
        row("pause", "UI Pause-badge check: write 200 short numbered sentences about geometric shapes. Do not use tools."),
        row("geometry", "1. A circle has no corners. 2. A triangle has three sides.", "assistant"),
    ],
    "music_stop_wrong_answer": [
        row("music", "آهنگ رو قطع کن"),
        row("professor", "The strongest professor at University of Toronto for LLM research is...", "assistant"),
    ],
    "elon_context": [
        row("elon", "How old is Elon Musk?"),
        row("age", "He was 55.", "assistant"),
        row("filler", "Hmm"),
        row("death", ". When will he die"),
        row("progress", "Let me check.", "assistant"),
        row("answer", "No one can know when Elon Musk will die.", "assistant"),
        row("food", "What do you think is the best food at Uber Eats?"),
    ],
    "persian_context": [
        row("entity", "ایلان ماسک چه کسی است؟"),
        row("transition", "اوکی، الان یه سؤال"),
        row("persian_age", "اون چند سالشه؟"),
    ],
    "professor_research": [
        row("research", "یه سرچ کن ببین قوی‌ترین استاد دانشگاه تورنتو توی بحث ال ال ام کیه؟"),
    ],
    "standalone_recap": [
        row("rick", "Tell me about Rick Rubin.", conversation="old"),
        row("weekly", "Weekly recap: completed work, open tasks and priorities for next week.",
            "assistant", conversation="routine", channel="routine"),
    ],
    "opening_greeting": [row("greeting", "Hi")],
}


def test_main_goal_excludes_irrelevant_assistant_and_keeps_aliases():
    for name in ("pause_control_test", "music_stop_wrong_answer"):
        rows = QUALITY_CASES[name]
        groups, aliases = topics._canonical_groups(rows)
        canonical = rows[0]["id"]
        assert set(groups) == {canonical}
        assert aliases[rows[1]["id"]] == canonical
        assert groups[canonical]["input"]["source_text"] == rows[0]["content"]
        assert rows[1]["content"] not in str(groups[canonical]["input"])


def test_elon_fillers_progress_replies_and_new_food_goal():
    groups, aliases = topics._canonical_groups(QUALITY_CASES["elon_context"])
    assert set(groups) == {"elon", "death", "food"}
    assert {aliases[id] for id in ("elon", "age", "filler")} == {"elon"}
    assert {aliases[id] for id in ("death", "progress", "answer")} == {"death"}
    assert groups["death"]["input"]["source_text"] == ". When will he die"
    assert groups["death"]["input"]["context_before"] == [{"role": "user", "content": "How old is Elon Musk?"}]
    assert "He was 55" not in str(groups["elon"]["input"])
    assert aliases["food"] == "food"


@pytest.mark.parametrize("filler", ["Okay, another question", "Hmm", "اوکی، الان یه سؤال", "اوكي، الان یه سُؤال", "باشه یه سوال دیگه"])
def test_fillers_do_not_replace_substantive_context(filler):
    rows = [row("entity", "Who is Elon Musk?"), row("filler", filler), row("followup", "And his education?")]
    groups, aliases = topics._canonical_groups(rows)
    assert aliases["filler"] == "entity"
    assert aliases["followup"] == "entity"
    assert groups["entity"]["input"]["source_text"] == "Who is Elon Musk?"
    assert filler not in groups["entity"]["input"]["followup_intents"]


def test_multilingual_new_goal_context_skips_transition():
    groups, aliases = topics._canonical_groups(QUALITY_CASES["persian_context"])
    assert aliases["transition"] == "entity"
    assert groups["persian_age"]["input"]["context_before"] == [{"role": "user", "content": "ایلان ماسک چه کسی است؟"}]
    rows = [row("entity", "Who is Elon Musk?"), row("filler", "Hmm"), row("age", "What is his age?")]
    groups, aliases = topics._canonical_groups(rows)
    assert aliases["age"] == "age"
    assert groups["age"]["input"]["context_before"] == [{"role": "user", "content": "Who is Elon Musk?"}]


@pytest.mark.parametrize("text", ["Show this week's sales summary", "Explain this new theorem: Fermat's Last Theorem", "Is this a good time to buy a house?", "Elon Musk age", "What do you think is the best food at Uber Eats?"])
def test_explicit_new_objects_do_not_merge_into_previous_goal(text):
    groups, aliases = topics._canonical_groups([row("music", "Stop music"), row("new", text)])
    assert topics._text_kind(text) == "new"
    assert set(groups) == {"music", "new"}
    assert aliases["new"] == "new"


@pytest.mark.parametrize("text", ["his age", "make it shorter", "show this again", "Can you show me in English?", "Okay, say it in English", "Could you translate that into Persian please?", "Say everything in English."])
def test_explicit_followups_keep_prior_goal(text):
    assert topics._text_kind(text) == "continuation"


def test_polite_language_switch_does_not_replace_elon_goal():
    rows = [row("entity", "ایلان ماسک"), row("age", "چند سالشه؟"),
            row("language", "Say everything in English."), row("ack", "Hmm"),
            row("death", "When will he die?"), row("progress", "Let me check.", "assistant")]
    groups, aliases = topics._canonical_groups(rows)
    assert {aliases[id] for id in ("age", "language", "ack")} == {"age"}
    assert aliases["death"] == aliases["progress"] == "death"
    assert groups["age"]["input"]["source_text"] == "چند سالشه؟"
    assert groups["age"]["input"]["context_before"] == [{"role": "user", "content": "ایلان ماسک"}]


def test_successive_short_questions_retain_named_user_context():
    rows = [row("entity", "ایلان ماسک"), row("age", "چند سالشه؟"),
            row("fragment", "شغلش چیه؟"), row("language", "Say everything in English."),
            row("death", "When will he die?")]
    groups, aliases = topics._canonical_groups(rows)
    assert aliases["language"] == "fragment"
    assert aliases["death"] == "death"
    assert groups["fragment"]["input"]["context_before"] == [
        {"role": "user", "content": "ایلان ماسک"}, {"role": "user", "content": "چند سالشه؟"}]
    assert groups["death"]["input"]["context_before"][0]["content"] == "ایلان ماسک"


@pytest.mark.parametrize("question", ["When will he die?", "What is his age?", "How old was he?", "Where does he work?"])
def test_clear_factual_question_is_own_goal_after_unclear_opener(question):
    rows = [row("entity", "Tell me about Elon Musk"), row("fragment", "unclear transcription fragment"),
            row("language", "Say everything in English."), row("question", question),
            row("progress", "Let me check.", "assistant"), row("answer", "Response to the question", "assistant")]
    groups, aliases = topics._canonical_groups(rows)
    assert aliases["language"] == "fragment"
    assert {aliases[id] for id in ("question", "progress", "answer")} == {"question"}
    assert groups["question"]["input"]["source_text"] == question
    assert groups["question"]["input"]["context_before"][0]["content"] == "Tell me about Elon Musk"
    assert len(groups["question"]["input"]["context_before"]) <= 6


def test_entity_survives_bounded_short_voice_fragment_context():
    rows = [row("entity", "ایلان ماسک")]
    rows += [row(f"fragment{i}", f"Transcription fragment {i}") for i in range(4)]
    rows += [row("age", "How old was he?"), row("death", "When will he die?")]
    groups, _ = topics._canonical_groups(rows)
    context = groups["death"]["input"]["context_before"]
    assert len(context) == 6
    assert context[0]["content"] == "ایلان ماسک"
    assert sum(len(item["content"]) for item in context) <= 2400


async def test_opaque_ids_use_compact_keys_and_map_back_exactly(monkeypatch):
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]

    async def model(**kwargs):
        inputs = json.loads(kwargs["messages"][0]["content"])
        assert [item["message_id"] for item in inputs] == ["t0", "t1"]
        assert all(id not in kwargs["messages"][0]["content"] for id in ids)
        return json.dumps({"topics": [{"message_id": "t0", "topic": "Test pause control"},
                                      {"message_id": "t1", "topic": "Stop playing music"},
                                      {"message_id": "t9", "topic": "Unrequested title"}]})

    monkeypatch.setattr("app.services.internal_llm.call_system_llm", model)
    result = await topics._generate("owner", [{"message_id": id, "source_text": "test"} for id in ids])
    assert result == {ids[0]: "Test pause control", ids[1]: "Stop playing music"}


def test_turn_identity_beats_intervening_question_and_routine_marker():
    rows = [row("first", "Stop music", client_msg_id="turn1"), row("second", "How old is Elon Musk?", client_msg_id="turn2"),
            row("delayed", "Stopped.", "assistant", client_msg_id="turn1", source="agent_task"),
            row("unmatched", "Unrelated update", "assistant", client_msg_id="unknown"),
            row("other", "Other conversation", "assistant", conversation="different", client_msg_id="turn1")]
    groups, aliases = topics._canonical_groups(rows)
    assert aliases["delayed"] == "first"
    assert aliases["unmatched"] == "unmatched"
    assert aliases["other"] == "other"
    assert groups["unmatched"]["input"]["context_before"] == []


def test_generic_agent_source_is_normal_reply_and_routine_is_standalone():
    rows = [row("user", "Stop music"), row("reply", "Stopped.", "assistant", source="agent"),
            row("routine", "Weekly work recap", "assistant", channel="routine")]
    groups, aliases = topics._canonical_groups(rows)
    assert aliases["reply"] == "user"
    assert aliases["routine"] == "routine"
    assert groups["routine"]["input"]["context_before"] == []


def test_voice_half_ids_use_same_conversation_chronology_without_text_id_fallback():
    def voice_id(name):
        return str(uuid.uuid5(uuid.NAMESPACE_OID, "toup-voice:session:" + name))
    rows = [row("voice-user", "How old is Elon Musk?", channel="voice", client_msg_id=voice_id("user")),
            row("voice-reply", "He was 55.", "assistant", channel="voice", client_msg_id=voice_id("reply")),
            row("voice-progress", "Let me check.", "assistant", channel="voice", client_msg_id=voice_id("progress")),
            row("text-reply", "Unmatched text turn", "assistant", client_msg_id=voice_id("text")),
            row("other-voice", "Other call", "assistant", conversation="other", channel="voice", client_msg_id=voice_id("other"))]
    groups, aliases = topics._canonical_groups(rows)
    assert aliases["voice-reply"] == aliases["voice-progress"] == "voice-user"
    assert aliases["text-reply"] == "text-reply"
    assert aliases["other-voice"] == "other-voice"
    assert "55" not in str(groups["voice-user"]["input"])


def test_persian_goal_gets_explicit_script_hint():
    groups, _ = topics._canonical_groups(QUALITY_CASES["music_stop_wrong_answer"])
    assert groups["music"]["input"]["title_language"] == "Persian (فارسی), in Persian script"
    assert "Arabic script" in topics._title_language("مرحبا بالعالم")
    assert "Latin script" in topics._title_language("Best food on Uber Eats")


def test_fingerprint_stable_across_reply_streaming_and_filler_but_not_intent():
    rows = [row("user", "Test stop control using geometry sentences"), row("reply", "A circle...", "assistant")]
    first = topics._canonical_groups(rows)[0]["user"]["fingerprint"]
    rows[1]["content"] += "\nA triangle has three sides."
    rows.append(row("filler", "Hmm"))
    assert topics._canonical_groups(rows)[0]["user"]["fingerprint"] == first
    rows.append(row("followup", "Make it shorter"))
    assert topics._canonical_groups(rows)[0]["user"]["fingerprint"] != first


def test_aliases_are_requested_id_keyed_and_bounded_to_proven_group():
    rows = [row("anchor", "Stop music")] + [row(f"reply{i}", "Done", "assistant") for i in range(30)]
    rows += [row("other", "How old is Elon Musk?", conversation="other")]
    groups, aliases = topics._canonical_groups(rows)
    result = topics._response(["reply29", "other"], groups, aliases, {"anchor": "Stop playing music"}, busy=True)
    assert result["retry_after_ms"] == 10000
    assert result["topics"][0]["message_id"] == "reply29"
    assert result["topics"][0]["canonical_message_id"] == "anchor"
    assert len(result["topics"][0]["alias_message_ids"]) == 20
    assert "other" not in result["topics"][0]["alias_message_ids"]


def test_timestamp_offsets_normalized_before_legacy_reply_pairing():
    assert topics._timestamp({"occurred_at": "2026-10-22T10:00:00-04:00"}) == datetime(2026, 10, 22, 14)
    rows = [row("user", "Stop music", occurred_at="2026-10-22T10:00:00-04:00"),
            row("early", "Old reply", "assistant", occurred_at="2026-10-22T13:59:00Z")]
    assert topics._canonical_groups(rows)[1]["early"] == "early"


@pytest.mark.parametrize('followup', ['کیه؟', 'کيه', 'چه کسیه؟', 'Who?', 'Who is that?'])
def test_bare_identity_followup_keeps_professor_goal_and_reply_aliases(followup):
    rows = [row('professor', 'قوی‌ترین استاد دانشگاه تورنتو توی بحث ال ال ام'),
            row('who', followup), row('progress', 'Let me check.', 'assistant')]
    groups, aliases = topics._canonical_groups(rows)
    assert set(groups) == {'professor'}
    assert {aliases[key] for key in ('professor', 'who', 'progress')} == {'professor'}
    assert groups['professor']['input']['source_text'] == rows[0]['content']
    assert groups['professor']['input']['followup_intents'] == [followup]
    assert 'Let me check' not in str(groups['professor']['input'])


@pytest.mark.parametrize('question', ['Who is Ada Lovelace?', 'ایلان ماسک کیه؟', 'چند سالشه؟', 'When will he die?'])
def test_explicit_identity_and_factual_questions_remain_separate(question):
    groups, aliases = topics._canonical_groups([row('professor', 'Find a professor researching LLMs'), row('new', question)])
    assert set(groups) == {'professor', 'new'}
    assert aliases['new'] == 'new'


def test_bare_identity_never_borrows_another_conversation():
    groups, aliases = topics._canonical_groups([row('professor', 'Find an LLM professor'), row('who', 'کیه', conversation='other')])
    assert aliases['who'] == 'who'
    assert groups['who']['input']['context_before'] == []
