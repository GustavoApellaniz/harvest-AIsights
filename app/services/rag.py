"""Orquestra o fluxo: recuperar trechos -> montar prompt -> chamar LLM -> guardar histórico."""

import logging
import time

from app.core.exceptions import IndexEmptyError
from app.services.history import ConversationStore
from app.services.llm import LLM
from app.services.prompts import SYSTEM_PROMPT, build_user_message
from app.services.vectorstore import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

NO_RELEVANT_ANSWER = "Não encontrei registros relevantes para responder a essa pergunta."


class RagService:
    def __init__(self, store: VectorStore, llm: LLM, history: ConversationStore, top_k: int, max_distance: float | None = None):
        self._store = store
        self._llm = llm
        self._history = history
        self._top_k = top_k
        self._max_distance = max_distance

    def answer(self, question: str, session_id: str) -> tuple[str, list[RetrievedChunk]]:
        started = time.perf_counter()
        # Perguntas de seguimento ("e o de Bordéus?") dependem do contexto anterior:
        # a busca usa a pergunta anterior + a atual (sem chamada extra ao LLM).
        previous = self._history.last_user_question(session_id)
        retrieval_query = f"{previous} {question}" if previous else question
        chunks = self._store.search(retrieval_query, self._top_k)
        if not chunks:
            raise IndexEmptyError(
                "Nenhum documento indexado. Coloque arquivos em data/ e rode: python -m app.ingest"
            )

        if self._max_distance is not None:
            chunks = [c for c in chunks if c.distance <= self._max_distance]
            if not chunks:
                # Havia documentos, mas nenhum relevante: responde sem chamar o LLM.
                self._history.add_turn(session_id, question, NO_RELEVANT_ANSWER)
                logger.info("session=%s sem trechos abaixo de max_distance=%s", session_id, self._max_distance)
                return NO_RELEVANT_ANSWER, []

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
