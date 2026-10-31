"""Leitura de documentos, divisão em trechos (chunks) e indexação."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.services.vectorstore import VectorStore

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".csv"}


@dataclass(frozen=True)
class Page:
    source: str
    page: int | None
    text: str
    row: int | None = None  # linha do CSV (cabeçalho = linha 1)


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    page: int | None
    text: str
    row: int | None = None


@dataclass
class IngestReport:
    files: int = 0
    chunks: int = 0
    skipped_empty: int = 0


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Divide texto em janelas de até `chunk_size` caracteres com `overlap` de sobreposição.

    Prefere quebrar em espaços para não cortar palavras. Quebras de linha são
    normalizadas para espaço (simplificação consciente: perde-se a estrutura de parágrafos).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser positivo")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap deve estar entre 0 e chunk_size - 1")

    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks: list[str] = []
    n = len(text)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            cut = text.rfind(" ", start + chunk_size // 2, end)
            if cut != -1:
                end = cut
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = max(end - overlap, start + 1)
        if text[nxt - 1] != " ":
            space = text.find(" ", nxt, end)
            if space != -1:
                nxt = space + 1
        start = nxt
    return chunks


def load_document(path: Path, base_dir: Path) -> list[Page]:
    """Lê um arquivo e devolve páginas (PDF) ou uma única 'página' (txt/md)."""
    source = path.relative_to(base_dir).as_posix()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            Page(source=source, page=i, text=page.extract_text() or "")
            for i, page in enumerate(reader.pages, start=1)
        ]
    if suffix == ".csv":
        from app.services.tabular import load_csv  # import local: tabular importa Page deste módulo

        return load_csv(path, base_dir)
    if suffix in {".txt", ".md"}:
        return [Page(source=source, page=None, text=path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"Formato não suportado: {suffix}")


def build_chunks(pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for idx, piece in enumerate(chunk_text(page.text, chunk_size, overlap)):
            if page.row:
                chunk_id = f"{page.source}::r{page.row}::c{idx}"
            else:
                chunk_id = f"{page.source}::p{page.page or 0}::c{idx}"
            chunks.append(Chunk(id=chunk_id, source=page.source, page=page.page, text=piece, row=page.row))
    return chunks


def iter_documents(data_dir: Path) -> list[Path]:
    return sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES)


def ingest_directory(data_dir: Path, store: VectorStore, chunk_size: int, overlap: int) -> IngestReport:
    """Indexa todos os documentos suportados. Reindexar um arquivo substitui seus trechos antigos."""
    report = IngestReport()
    for path in iter_documents(data_dir):
        pages = load_document(path, data_dir)
        chunks = build_chunks(pages, chunk_size, overlap)
        if not chunks:
            logger.warning("Nenhum texto extraído de %s (PDF escaneado? OCR não é suportado).", path)
            report.skipped_empty += 1
            continue
        store.replace_source(chunks[0].source, chunks)
        report.files += 1
        report.chunks += len(chunks)
        logger.info("Indexado %s: %d trechos", chunks[0].source, len(chunks))
    return report
