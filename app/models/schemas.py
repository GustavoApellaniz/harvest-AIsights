"""Contratos da API (validação de entrada e formato de saída)."""

from pydantic import BaseModel, Field, field_validator


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{1,64}$",
        description="Identifica a conversa. Se omitido, um novo id é gerado.",
    )

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("A pergunta não pode estar vazia.")
        return value


class SourceChunk(BaseModel):
    ref: int = Field(description="Número usado nas citações [n] da resposta.")
    source: str
    page: int | None = Field(default=None, description="Página (PDF).")
    row: int | None = Field(default=None, description="Linha do arquivo CSV (cabeçalho = linha 1).")
    chunk_id: str
    distance: float = Field(description="Distância de cosseno (menor = mais similar).")
    excerpt: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceChunk] = Field(
        description="Trechos recuperados e enviados ao LLM (não necessariamente todos citados)."
    )


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
