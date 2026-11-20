from app.services.ingestion import ingest_directory
from app.services.rag import NO_RELEVANT_ANSWER, RagService


def _index(tmp_path, store):
    data = tmp_path / "data"
    data.mkdir()
    (data / "milho.txt").write_text("O milho precisa de solo bem drenado.", encoding="utf-8")
    ingest_directory(data, store, 800, 100)


def test_followup_uses_previous_question_for_retrieval_only(tmp_path, store, fake_llm, history):
    _index(tmp_path, store)
    queries = []
    original = store.search
    store.search = lambda q, k: (queries.append(q), original(q, k))[1]
    rag = RagService(store, fake_llm, history, top_k=3)

    rag.answer("milho solo", "s")
    rag.answer("e a drenagem?", "s")

    assert queries[0] == "milho solo"
    assert queries[1] == "milho solo e a drenagem?"
    assert "milho solo" not in fake_llm.calls[1]["messages"][-1]["content"]


def test_max_distance_filters_all_without_calling_llm(tmp_path, store, fake_llm, history):
    _index(tmp_path, store)
    rag = RagService(store, fake_llm, history, top_k=3, max_distance=0.0)
    answer, sources = rag.answer("assunto totalmente diferente", "s")
    assert answer == NO_RELEVANT_ANSWER and sources == []
    assert fake_llm.calls == []
    assert history.get_messages("s")[-1]["content"] == NO_RELEVANT_ANSWER


def test_max_distance_none_keeps_chunks(tmp_path, store, fake_llm, history):
    _index(tmp_path, store)
    _, sources = RagService(store, fake_llm, history, top_k=3).answer("milho", "s")
    assert sources and fake_llm.calls
