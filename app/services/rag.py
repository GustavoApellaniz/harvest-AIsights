"""Orquestra o fluxo: recuperar trechos -> montar prompt -> chamar LLM -> guardar histórico."""

import logging
import time

from app.core.exceptions import IndexEmptyError
from app.services.history import ConversationStore
from app.services.llm import LLM
from app.services.prompts import SYSTEM_PROMPT, build_user_message
from app.services.vectorstore import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class RagService:
    def __init__(self, store: VectorStore, llm: LLM, history: ConversationStore, top_k: int):
        self._store = store
        self._llm = llm
        self._history = history
        self._top_k = top_k

    def answer(self, question: str, session_id: str) -> tuple[str, list[RetrievedChunk]]:
        started = time.perf_counter()
        # A busca usa apenas a pergunta atual (limitação: perguntas de seguimento
        # como "e sobre o item anterior?" não são reescritas com o histórico).
        chunks = self._store.search(question, self._top_k)
        if not chunks:
            raise IndexEmptyError(
                "Nenhum documento indexado. Coloque arquivos em data/ e rode: python -m app.ingest"
            )

        messages = self._history.get_messages(session_id)
        messages.append({"role": "user", "content": build_user_message(question, chunks)})
        answer = self._llm.generate(SYSTEM_PROMPT, messages)
        self._history.add_turn(session_id, question, answer)

        logger.info(
            "session=%s question_chars=%d retrieved=%d answer_chars=%d latency_ms=%d",
            session_id,
            len(question),
            len(chunks),
            len(answer),
            (time.perf_counter() - started) * 1000,
        )
        return answer, chunks

    def reset_session(self, session_id: str) -> None:
        self._history.clear(session_id)
