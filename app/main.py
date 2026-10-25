"""FastAPI endpoints for chat, document ingestion, and service health."""

import re
import secrets
import threading
import time
from pathlib import Path
from typing import Literal
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import BaseModel, Field, field_validator
from qdrant_client.http.exceptions import ApiException
from urllib.parse import urlsplit

from src.config import get_settings
from src.conversation.router import Intent, route_message
from src.generation.answer import generate_answer
from src.guardrails.cache import SemanticCache
from src.guardrails.validation import is_grounded, validate_input
from src.ingestion.chunking import chunk_file
from src.retrieval.index import _client, index_chunks
from src.retrieval.retriever import HybridRetriever

app = FastAPI(title=get_settings().app_name, version="0.1.0")
ADMIN_SESSION_TTL_SECONDS = 1800
ADMIN_LOGIN_WINDOW_SECONDS = 900
ADMIN_LOGIN_MAX_ATTEMPTS = 5
_password_hasher = PasswordHasher()
_login_lock = threading.Lock()
_failed_admin_logins: dict[str, list[float]] = {}


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


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024, repr=False)


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


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


def _admin_token_serializer() -> URLSafeTimedSerializer:
    secret = get_settings().admin_session_secret
    if not secret:
        raise HTTPException(
            status_code=503, detail="Admin authentication is not configured"
        )
    return URLSafeTimedSerializer(secret, salt="admin-session-v1")


def verify_admin_token(
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> dict[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid or expired admin session")
    serializer = _admin_token_serializer()
    try:
        claims = serializer.loads(
            authorization.removeprefix("Bearer "),
            max_age=ADMIN_SESSION_TTL_SECONDS,
        )
    except BadSignature as error:
        raise HTTPException(
            status_code=401, detail="Invalid or expired admin session"
        ) from error
    settings = get_settings()
    if (
        not settings.admin_username
        or claims.get("role") != "admin"
        or claims.get("sub") != settings.admin_username
    ):
        raise HTTPException(status_code=401, detail="Invalid or expired admin session")
    return claims


def _login_attempts_remaining(client_key: str, now: float) -> int:
    with _login_lock:
        for key in list(_failed_admin_logins):
            recent = [
                attempt
                for attempt in _failed_admin_logins[key]
                if now - attempt < ADMIN_LOGIN_WINDOW_SECONDS
            ]
            if recent:
                _failed_admin_logins[key] = recent
            else:
                del _failed_admin_logins[key]
        return ADMIN_LOGIN_MAX_ATTEMPTS - len(_failed_admin_logins.get(client_key, []))


def _record_failed_admin_login(client_key: str, now: float) -> None:
    with _login_lock:
        _failed_admin_logins.setdefault(client_key, []).append(now)


def _clear_failed_admin_logins(client_key: str) -> None:
    with _login_lock:
        _failed_admin_logins.pop(client_key, None)


@app.post("/admin/login", response_model=AdminLoginResponse)
def admin_login(
    credentials: AdminLoginRequest,
    request: Request,
    _: None = Depends(verify_api_key),
) -> AdminLoginResponse:
    settings = get_settings()
    if not all(
        (
            settings.admin_username,
            settings.admin_password_hash,
            settings.admin_session_secret,
        )
    ):
        raise HTTPException(
            status_code=503, detail="Admin authentication is not configured"
        )

    client_key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    if _login_attempts_remaining(client_key, now) <= 0:
        raise HTTPException(
            status_code=429,
            detail="Too many failed login attempts. Try again later.",
        )

    username_matches = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        settings.admin_username.encode("utf-8"),
    )
    try:
        password_matches = _password_hasher.verify(
            settings.admin_password_hash, credentials.password
        )
    except VerifyMismatchError:
        password_matches = False
    except (InvalidHashError, VerificationError) as error:
        raise HTTPException(
            status_code=503, detail="Admin password hash configuration is invalid"
        ) from error

    if not username_matches or not password_matches:
        _record_failed_admin_login(client_key, now)
        raise HTTPException(status_code=401, detail="Invalid username or password")

    _clear_failed_admin_logins(client_key)
    token = _admin_token_serializer().dumps(
        {"sub": settings.admin_username, "role": "admin"}
    )
    return AdminLoginResponse(
        access_token=token,
        expires_in=ADMIN_SESSION_TTL_SECONDS,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _qdrant_status() -> dict[str, str | int | bool]:
    collection_name = get_settings().qdrant_collection
    try:
        client = _client()
        collections = client.get_collections().collections
        collection_exists = any(item.name == collection_name for item in collections)
        count = (
            client.count(collection_name=collection_name, exact=True).count
            if collection_exists
            else 0
        )
    except ApiException as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve ingestion status from Qdrant",
        ) from error
    return {
        "status": "active",
        "collection_name": collection_name,
        "collection_exists": collection_exists,
        "total_chunks_ingested": count,
        "has_documents": count > 0,
    }


@app.get("/status")
def get_ingestion_status(
    _: None = Depends(verify_api_key),
) -> dict[str, str | int | bool]:
    return _qdrant_status()


@app.get("/admin/status")
def get_admin_status(
    _: dict[str, str] = Depends(verify_admin_token),
) -> dict[str, str | int | bool]:
    return _qdrant_status()


@app.get("/admin/config")
def get_admin_config(
    _: dict[str, str] = Depends(verify_admin_token),
) -> dict[str, str | bool]:
    settings = get_settings()
    qdrant_host = urlsplit(
        settings.qdrant_url
        if "://" in settings.qdrant_url
        else f"https://{settings.qdrant_url}"
    ).hostname
    llm_host = urlsplit(settings.local_model_base_url).hostname
    return {
        "qdrant_host": qdrant_host or "local embedded",
        "qdrant_collection": settings.qdrant_collection,
        "qdrant_api_key_configured": bool(settings.qdrant_api_key),
        "llm_host": llm_host or "not configured",
        "llm_model": settings.local_model_name,
        "llm_api_key_configured": bool(settings.local_model_api_key),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "api_access_key_configured": bool(settings.api_access_key),
    }


@app.get("/admin/chunks")
def get_admin_chunks(
    offset: str | None = Query(default=None, max_length=64),
    limit: int = Query(default=50, ge=1, le=100),
    _: dict[str, str] = Depends(verify_admin_token),
) -> dict[str, object]:
    settings = get_settings()
    status = _qdrant_status()
    if not status["collection_exists"]:
        return {"chunks": [], "next_page_offset": None}
    try:
        points, next_page_offset = _client().scroll(
            collection_name=settings.qdrant_collection,
            limit=limit,
            offset=offset,
            with_payload=["source", "chunk_index", "page_number"],
            with_vectors=False,
        )
    except ApiException as error:
        raise HTTPException(
            status_code=503,
            detail="Unable to retrieve chunk details from Qdrant",
        ) from error

    chunks = []
    for point in points:
        payload = point.payload or {}
        source = str(payload.get("source") or "Unknown document")
        filename = source.replace("\\", "/").rsplit("/", maxsplit=1)[-1]
        filename = re.sub(r"^[0-9a-fA-F]{32}_", "", filename)
        chunks.append(
            {
                "document": filename,
                "chunk_index": payload.get("chunk_index"),
                "page_number": payload.get("page_number"),
            }
        )
    return {
        "chunks": chunks,
        "next_page_offset": str(next_page_offset) if next_page_offset else None,
    }


@app.post("/admin/ingest", response_model=IngestResponse)
async def admin_ingest(
    files: list[UploadFile] = File(...),
    _: dict[str, str] = Depends(verify_admin_token),
) -> IngestResponse:
    return await _ingest_files(files)


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
    return await _ingest_files(files)


async def _ingest_files(files: list[UploadFile]) -> IngestResponse:
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
