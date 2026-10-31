from app.services.history import ConversationStore


def test_messages_alternate_and_start_with_user():
    h = ConversationStore(max_turns=3, max_sessions=10)
    h.add_turn("s", "q1", "a1")
    h.add_turn("s", "q2", "a2")
    msgs = h.get_messages("s")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[0]["content"] == "q1"


def test_only_last_turns_are_kept():
    h = ConversationStore(max_turns=2, max_sessions=10)
    for i in range(4):
        h.add_turn("s", f"q{i}", f"a{i}")
    assert [m["content"] for m in h.get_messages("s")] == ["q2", "a2", "q3", "a3"]


def test_least_recently_used_session_is_evicted():
    h = ConversationStore(max_turns=2, max_sessions=2)
    h.add_turn("a", "q", "r")
    h.add_turn("b", "q", "r")
    h.get_messages("a")  # 'a' vira a mais recente
    h.add_turn("c", "q", "r")
    assert h.get_messages("b") == []
    assert h.get_messages("a") != []


def test_clear_and_zero_turns():
    h = ConversationStore(max_turns=2, max_sessions=2)
    h.add_turn("a", "q", "r")
    h.clear("a")
    assert h.get_messages("a") == []
    off = ConversationStore(max_turns=0, max_sessions=2)
    off.add_turn("a", "q", "r")
    assert off.get_messages("a") == []
