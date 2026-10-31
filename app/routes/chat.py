from uuid import uuid4

from fastapi import APIRouter, Depends, Response, status

from app.dependencies import get_rag_service
from app.models.schemas import ChatRequest, ChatResponse, SourceChunk
from app.services.rag import RagService

router = APIRouter(prefix="/chat", tags=["chat"])

EXCERPT_CHARS = 240


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, rag: RagService = Depends(get_rag_service)) -> ChatResponse:
    session_id = request.session_id or uuid4().hex
    answer, chunks = rag.answer(request.question, session_id)
    sources = [
        SourceChunk(
            ref=ref,
            source=c.source,
            page=c.page,
            row=c.row,
            chunk_id=c.chunk_id,
            distance=round(c.distance, 4),
            excerpt=c.text[:EXCERPT_CHARS],
        )
        for ref, c in enumerate(chunks, start=1)
    ]
    return ChatResponse(session_id=session_id, answer=answer, sources=sources)


@router.delete("/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def reset_session(session_id: str, rag: RagService = Depends(get_rag_service)) -> Response:
    rag.reset_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
