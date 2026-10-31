from eval.evaluate import evaluate, is_hit
from app.services.vectorstore import RetrievedChunk


def _c(source, text="t"):
    return RetrievedChunk(chunk_id="i", source=source, page=None, text=text, distance=0.0)


def test_hit_rate_and_mrr():
    cases = [
        {"question": "q1", "expected_source": "a.pdf"},
        {"question": "q2", "expected_source": "b.pdf"},
        {"question": "q3", "expected_source": "c.pdf", "expected_text": "Dose   Máxima"},
    ]
    results = {
        "q1": [_c("a.pdf")],
        "q2": [_c("x.pdf"), _c("b.pdf")],
        "q3": [_c("c.pdf", "a dose máxima é 5 mg")],
    }
    out = evaluate(cases, lambda q, k: results[q][:k], k=2)
    assert out["hit_rate"] == 1.0
    assert out["mrr"] == (1 + 0.5 + 1) / 3

    out_k1 = evaluate(cases, lambda q, k: results[q][:k], k=1)
    assert out_k1["misses"] == ["q2"]


def test_expected_text_must_match():
    assert not is_hit({"question": "q", "expected_source": "a", "expected_text": "xyz"}, _c("a", "abc"))
