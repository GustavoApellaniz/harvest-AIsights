import pytest

from app.services.ingestion import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("   \n\t ", 100, 10) == []


def test_short_text_is_single_chunk():
    assert chunk_text("olá  mundo\nteste", 100, 10) == ["olá mundo teste"]


def test_long_text_respects_size_overlap_and_coverage():
    words = [f"palavra{i}" for i in range(200)]
    chunks = chunk_text(" ".join(words), chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    covered = {w for c in chunks for w in c.split()}
    assert covered == set(words)
    for prev, nxt in zip(chunks, chunks[1:]):
        assert prev.split()[-1] in nxt.split()  # sobreposição


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("abc", 10, 10)


def test_zero_overlap_terminates():
    chunks = chunk_text("a " * 500, 50, 0)
    assert len(chunks) > 1
