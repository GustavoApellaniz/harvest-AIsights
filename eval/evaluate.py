"""Avalia apenas a recuperação (retrieval): o trecho certo aparece nos top-k resultados?

Uso: python -m eval.evaluate --questions eval/questions.jsonl --k 4

Cada linha do JSONL: {"question": "...", "expected_source": "arquivo.pdf", "expected_text": "trecho opcional"}
- expected_source: caminho relativo do documento dentro de data/ (igual ao campo `source` da API).
- expected_row (opcional, CSV): número da linha no arquivo (cabeçalho = linha 1), igual ao campo `row` da API.
- expected_text (opcional): texto que deve aparecer em algum trecho recuperado. A comparação ignora
  maiúsculas e espaços repetidos; se o texto atravessar a fronteira entre dois trechos, será contado como erro.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Callable

from app.services.vectorstore import RetrievedChunk


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_hit(case: dict, chunk: RetrievedChunk) -> bool:
    if chunk.source != case["expected_source"]:
        return False
    if case.get("expected_row") is not None and chunk.row != case["expected_row"]:
        return False
    expected_text = case.get("expected_text")
    return expected_text is None or _norm(expected_text) in _norm(chunk.text)


def evaluate(cases: list[dict], search: Callable[[str, int], list[RetrievedChunk]], k: int) -> dict:
    hits = 0
    reciprocal_ranks = 0.0
    misses = []
    for case in cases:
        results = search(case["question"], k)
        rank = next((i for i, c in enumerate(results, start=1) if is_hit(case, c)), None)
        if rank is None:
            misses.append(case["question"])
        else:
            hits += 1
            reciprocal_ranks += 1 / rank
    total = len(cases)
    return {
        "total": total,
        "k": k,
        "hit_rate": hits / total if total else 0.0,
        "mrr": reciprocal_ranks / total if total else 0.0,
        "misses": misses,
    }


def load_cases(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    from app.core.config import get_settings
    from app.services.embeddings import SentenceTransformerEmbedder
    from app.services.vectorstore import VectorStore

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.top_k
    store = VectorStore(settings.chroma_path, settings.collection_name, SentenceTransformerEmbedder(settings.embedding_model))
    result = evaluate(load_cases(args.questions), store.search, k)
    print(f"Perguntas: {result['total']} | hit@{k}: {result['hit_rate']:.1%} | MRR: {result['mrr']:.3f}")
    for question in result["misses"]:
        print(f"  - sem acerto: {question}")


if __name__ == "__main__":
    main()
