from app.services.ingestion import build_chunks, ingest_directory, iter_documents, load_document
from tests.conftest import make_pdf_bytes


def _write_docs(data_dir):
    (data_dir / "sub").mkdir(parents=True)
    (data_dir / "solo.txt").write_text("O milho precisa de solo bem drenado e adubação nitrogenada.", encoding="utf-8")
    (data_dir / "sub" / "leite.md").write_text("# Leite\nA pasteurização reduz microrganismos patogênicos no leite.", encoding="utf-8")
    (data_dir / "relatorio.pdf").write_bytes(make_pdf_bytes(["Pagina um sobre trigo", "Pagina dois sobre cevada"]))
    (data_dir / "ignorado.docx").write_bytes(b"nao suportado")


def test_iter_documents_only_supported(tmp_path):
    _write_docs(tmp_path)
    names = [p.name for p in iter_documents(tmp_path)]
    assert sorted(names) == ["leite.md", "relatorio.pdf", "solo.txt"]


def test_pdf_pages_are_loaded_with_page_numbers(tmp_path):
    _write_docs(tmp_path)
    pages = load_document(tmp_path / "relatorio.pdf", tmp_path)
    assert [p.page for p in pages] == [1, 2]
    assert "trigo" in pages[0].text and "cevada" in pages[1].text


def test_chunk_ids_are_unique_and_deterministic(tmp_path):
    _write_docs(tmp_path)
    pages = load_document(tmp_path / "sub" / "leite.md", tmp_path)
    a = build_chunks(pages, 50, 10)
    b = build_chunks(pages, 50, 10)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)
    assert a[0].source == "sub/leite.md"


def test_ingest_and_search_finds_right_source(tmp_path, store):
    data = tmp_path / "data"
    data.mkdir()
    _write_docs(data)
    report = ingest_directory(data, store, 800, 100)
    assert report.files == 3 and report.chunks >= 3

    result = store.search("pasteurização do leite", 1)
    assert result[0].source == "sub/leite.md"
    result = store.search("cevada", 1)
    assert (result[0].source, result[0].page) == ("relatorio.pdf", 2)


def test_reingest_is_idempotent_and_replaces_changed_files(tmp_path, store):
    data = tmp_path / "data"
    data.mkdir()
    _write_docs(data)
    ingest_directory(data, store, 800, 100)
    first = store.count()
    ingest_directory(data, store, 800, 100)
    assert store.count() == first

    (data / "solo.txt").write_text("Texto totalmente novo sobre soja.", encoding="utf-8")
    ingest_directory(data, store, 800, 100)
    assert store.count() == first
    assert all("milho" not in c.text for c in store.search("milho solo", 10))


def test_cosine_distance_is_zero_for_identical_and_one_for_unrelated(tmp_path, store):
    data = tmp_path / "data"
    data.mkdir()
    (data / "a.txt").write_text("alfa beta gama", encoding="utf-8")
    ingest_directory(data, store, 800, 100)
    same = store.search("alfa beta gama", 1)[0]
    other = store.search("zzzz yyyy", 1)[0]
    assert same.distance < 0.01
    assert other.distance > 0.9


def test_empty_pdf_text_is_skipped(tmp_path, store):
    data = tmp_path / "data"
    data.mkdir()
    (data / "vazio.txt").write_text("   ", encoding="utf-8")
    report = ingest_directory(data, store, 800, 100)
    assert report.files == 0 and report.skipped_empty == 1
    assert store.search("qualquer", 3) == []
