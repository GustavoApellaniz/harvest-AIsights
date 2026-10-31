"""Construção (uma única vez) dos serviços usados pelas rotas."""

from functools import lru_cache

from app.core.config import get_settings
from app.services.embeddings import SentenceTransformerEmbedder
from app.services.history import ConversationStore
from app.services.llm import GeminiLLM
from app.services.rag import RagService
from app.services.vectorstore import VectorStore


@lru_cache
def get_rag_service() -> RagService:
    settings = get_settings()
    api_key = settings.gemini_api_key.get_secret_value() if settings.gemini_api_key else None
    llm = GeminiLLM(
        api_key=api_key,
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
    )
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    store = VectorStore(settings.chroma_path, settings.collection_name, embedder)
    history = ConversationStore(settings.max_history_turns, settings.max_sessions)
    return RagService(store, llm, history, settings.top_k)
