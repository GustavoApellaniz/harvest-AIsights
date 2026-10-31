"""Geração de embeddings (vetores) com modelos da biblioteca sentence-transformers."""

import threading
from typing import Protocol


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Carrega o modelo na primeira utilização (evita custo de startup e de testes).

    Modelos da família E5 exigem os prefixos "query: " e "passage: ".
    """

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name)
            return self._model

    def _uses_e5_prefixes(self) -> bool:
        return "e5" in self._model_name.lower()

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._uses_e5_prefixes():
            texts = [f"passage: {t}" for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        if self._uses_e5_prefixes():
            text = f"query: {text}"
        return self._encode([text])[0]
