"""Acesso ao ChromaDB. Os embeddings são calculados por nós e passados explicitamente."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.services.embeddings import Embedder

if TYPE_CHECKING:
    from app.services.ingestion import Chunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    source: str
    page: int | None
    text: str
    distance: float
    row: int | None = None


class VectorStore:
    def __init__(self, path: Path, collection_name: str, embedder: Embedder):
        self._embedder = embedder
        client = chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self._collection.count()

    def replace_source(self, source: str, chunks: "list[Chunk]") -> None:
        """Remove os trechos antigos de `source` e insere os novos (idempotente)."""
        self._collection.delete(where={"source": source})
        if not chunks:
            return
        texts = [c.text for c in chunks]
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=self._embedder.embed_documents(texts),
            documents=texts,
            metadatas=[{"source": c.source, "page": c.page or 0, "row": c.row or 0} for c in chunks],
        )

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        total = self.count()
        if total == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(k, total),
            include=["documents", "metadatas", "distances"],
        )
        retrieved = []
        for chunk_id, text, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source=meta["source"],
                    page=meta["page"] or None,
                    row=meta.get("row") or None,
                    text=text,
                    distance=float(dist),
                )
            )
        return retrieved
