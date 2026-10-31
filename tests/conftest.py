import re
import zlib

import pytest

from app.core.exceptions import LLMError
from app.services.history import ConversationStore
from app.services.rag import RagService
from app.services.vectorstore import VectorStore


class FakeEmbedder:
    """Embedding determinístico por 'saco de palavras' (hash). Não é semântico, só serve para testes."""

    DIM = 256

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for token in re.findall(r"\w+", text.lower()):
            vec[zlib.crc32(token.encode()) % self.DIM] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self, reply: str = "Resposta de teste [1]", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[dict] = []

    def generate(self, system, messages):
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        if self.error:
            raise self.error
        return self.reply


def make_pdf_bytes(pages: list[str]) -> bytes:
    """PDF mínimo escrito à mão (uma linha de texto por página; sem parênteses no texto)."""
    n = len(pages)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        content_id = 5 + 2 * i
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 3 0 R >> >> >>"
            ).encode()
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "chroma", "test", FakeEmbedder())


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def history():
    return ConversationStore(max_turns=5, max_sessions=100)


@pytest.fixture
def rag(store, fake_llm, history):
    return RagService(store, fake_llm, history, top_k=3)


@pytest.fixture
def llm_error():
    return LLMError("falha simulada")
