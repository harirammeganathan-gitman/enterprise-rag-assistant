"""FastAPI endpoints for chat, document ingestion, and service health."""

import secrets
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from src.config import get_settings
from src.conversation.router import Intent, route_message
from src.generation.answer import generate_answer
from src.guardrails.cache import SemanticCache
from src.guardrails.validation import is_grounded, validate_input
from src.ingestion.chunking import chunk_file
from src.retrieval.index import index_chunks
from src.retrieval.retriever import HybridRetriever

app = FastAPI(title=get_settings().app_name, version="0.1.0")


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)

    @field_validator("history")
    @classmethod
    def validate_history_size(cls, history: list[ChatTurn]) -> list[ChatTurn]:
        total_characters = sum(len(turn.content) for turn in history)
        if total_characters > 8000:
            raise ValueError("Conversation history must not exceed 8000 characters")
        return history


class Citation(BaseModel):
    source: str
    page_number: int | None
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cached: bool = False


class IngestResponse(BaseModel):
    files_processed: int
    chunks_indexed: int


def _document_history(history: list[ChatTurn]) -> list[dict[str, str]]:
    relevant: list[dict[str, str]] = []
    pending_user: ChatTurn | None = None
    for turn in history:
        if turn.role == "user":
            pending_user = turn
        elif pending_user is not None:
            intent = route_message(
                pending_user.content, has_history=bool(relevant)
            ).intent
            if intent in {Intent.DOCUMENT_QUESTION, Intent.FOLLOW_UP}:
                relevant.extend(
                    [
                        {"role": "user", "content": pending_user.content},
                        {"role": "assistant", "content": turn.content},
                    ]
                )
            pending_user = None
    return relevant[-8:]


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    settings = get_settings()
    expected_key = settings.api_access_key
    if not expected_key:
        if settings.require_api_key:
            raise HTTPException(
                status_code=503, detail="API access key is not configured"
            )
        return
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode("utf-8"), expected_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, _: None = Depends(verify_api_key)) -> ChatResponse:
    try:
        validate_input(request.query)
        for turn in request.history:
            if turn.role == "user":
                validate_input(turn.content)
        history = _document_history(request.history)
        route = route_message(request.query, has_history=bool(history))
        if route.response:
            return ChatResponse(answer=route.response, citations=[])

        retrieval_query = request.query
        if route.intent == Intent.FOLLOW_UP:
            previous_user = next(
                (
                    turn["content"]
                    for turn in reversed(history)
                    if turn["role"] == "user"
                ),
                "",
            )
            previous_assistant = next(
                (
                    turn["content"]
                    for turn in reversed(history)
                    if turn["role"] == "assistant"
                ),
                "",
            )
            retrieval_query = (
                f"Previous question: {previous_user}\n"
                f"Previous answer: {previous_assistant}\n"
                f"Follow-up: {request.query}"
            )

        cache = None
        cached = None
        if route.intent != Intent.FOLLOW_UP:
            cache = SemanticCache()
            cached = cache.get(request.query)
        if cached is not None:
            return ChatResponse(
                answer=cached["answer"], citations=cached["citations"], cached=True
            )
        results = HybridRetriever().retrieve(retrieval_query, request.top_k)
        if not results:
            raise HTTPException(
                status_code=404, detail="No indexed documents are available"
            )
        contexts = [item["text"] for item in results]
        answer = generate_answer(
            request.query,
            contexts,
            history=history if route.intent == Intent.FOLLOW_UP else None,
        )
        if not is_grounded(answer, contexts):
            raise HTTPException(
                status_code=502,
                detail="Generated answer did not pass the context grounding check",
            )
        citations = [
            Citation(
                source=item["source"],
                page_number=item.get("page_number"),
                chunk_index=item["chunk_index"],
                score=item["rerank_score"],
            )
            for item in results
        ]
        response = ChatResponse(answer=answer, citations=citations)
        if cache is not None:
            cache.put(request.query, response.model_dump())
        return response
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    files: list[UploadFile] = File(...),
    _: None = Depends(verify_api_key),
) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="At least one file is required")
    upload_dir = get_settings().data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    processed = 0
    for upload in files:
        filename = Path(upload.filename or "").name
        if Path(filename).suffix.lower() not in {".pdf", ".docx", ".md", ".markdown"}:
            raise HTTPException(
                status_code=415, detail=f"Unsupported file type: {filename}"
            )
        content = await upload.read()
        if not content:
            raise HTTPException(
                status_code=400, detail=f"Uploaded file is empty: {filename}"
            )
        destination = upload_dir / f"{uuid4().hex}_{filename}"
        destination.write_bytes(content)
        chunks.extend(chunk_file(destination))
        processed += 1
    indexed = index_chunks(chunks)
    SemanticCache().clear()
    return IngestResponse(files_processed=processed, chunks_indexed=indexed)
