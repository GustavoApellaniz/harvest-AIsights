"""Histórico de conversas em memória (perdido ao reiniciar o servidor)."""

import threading
from collections import OrderedDict, deque


class ConversationStore:
    """Guarda os últimos `max_turns` pares (pergunta, resposta) por sessão.

    Mantém no máximo `max_sessions` sessões; ao exceder, descarta a menos usada recentemente.
    Só a pergunta original e a resposta são guardadas, não os trechos recuperados,
    para o prompt não crescer a cada turno.
    """

    def __init__(self, max_turns: int, max_sessions: int):
        self._max_turns = max_turns
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[tuple[str, str]]] = OrderedDict()
        self._lock = threading.Lock()

    def get_messages(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            turns = self._sessions.get(session_id)
            if turns is None:
                return []
            self._sessions.move_to_end(session_id)
            messages: list[dict[str, str]] = []
            for question, answer in turns:
                messages.append({"role": "user", "content": question})
                messages.append({"role": "assistant", "content": answer})
            return messages

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        if self._max_turns == 0:
            return
        with self._lock:
            turns = self._sessions.setdefault(session_id, deque(maxlen=self._max_turns))
            turns.append((question, answer))
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
