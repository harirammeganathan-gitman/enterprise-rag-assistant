"""FastAPI endpoints for chat, document ingestion, and service health."""

import secrets
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import get_settings
from src.generation.answer import generate_answer
from src.guardrails.cache import SemanticCache
from src.guardrails.validation import is_grounded, validate_input
from src.ingestion.chunking import chunk_file
from src.retrieval.index import index_chunks
from src.retrieval.retriever import HybridRetriever

app = FastAPI(title=get_settings().app_name, version="0.1.0")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)


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
        cache = SemanticCache()
        cached = cache.get(request.query)
        if cached:
            return ChatResponse(
                answer=cached["answer"], citations=cached["citations"], cached=True
            )
        results = HybridRetriever().retrieve(request.query, request.top_k)
        if not results:
            raise HTTPException(
                status_code=404, detail="No indexed documents are available"
            )
        contexts = [item["text"] for item in results]
        answer = generate_answer(request.query, contexts)
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
