"""CLI de indexação: python -m app.ingest [--data-dir DIR]"""

import argparse
import logging
from pathlib import Path

from app.core.config import get_settings
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.ingestion import ingest_directory
from app.services.vectorstore import VectorStore


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Indexa documentos (.pdf, .txt, .md, .csv) no ChromaDB.")
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    args = parser.parse_args()

    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    if not args.data_dir.is_dir():
        raise SystemExit(f"Pasta não encontrada: {args.data_dir}")

    store = VectorStore(settings.chroma_path, settings.collection_name, SentenceTransformerEmbedder(settings.embedding_model))
    report = ingest_directory(args.data_dir, store, settings.chunk_size, settings.chunk_overlap)
    print(f"Arquivos indexados: {report.files} | trechos: {report.chunks} | sem texto: {report.skipped_empty}")
    print(f"Total de trechos no índice: {store.count()}")


if __name__ == "__main__":
    main()
