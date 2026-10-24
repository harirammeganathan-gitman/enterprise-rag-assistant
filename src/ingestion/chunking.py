"""Recursive text splitting with token-estimated size and overlap."""

import re
from pathlib import Path

from src.ingestion.loaders import load_document
from src.ingestion.models import DocumentChunk

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SEPARATORS = ("\n\n", "\n", ". ", " ", "")


def _token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _token_count(text) <= chunk_size:
        return [text]

    for separator in SEPARATORS:
        if separator and separator not in text:
            continue
        pieces = text.split(separator) if separator else list(text)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current}{separator}{piece}" if current else piece
            if current and _token_count(candidate) > chunk_size:
                chunks.append(current.strip())
                tail = TOKEN_PATTERN.findall(current)
                current = " ".join(tail[-overlap:]) if overlap else ""
                if _token_count(f"{current} {piece}") > chunk_size:
                    if separator:
                        subchunks = _split_text(piece, chunk_size, overlap)
                        chunks.extend(subchunks[:-1])
                        current = subchunks[-1] if subchunks else ""
                    else:
                        current = piece
                else:
                    current = f"{current}{separator}{piece}".strip()
            else:
                current = candidate
        if current.strip():
            chunks.append(current.strip())
        if all(_token_count(chunk) <= chunk_size for chunk in chunks):
            return chunks
    raise ValueError("Unable to split text within the requested chunk size")


def chunk_file(
    path: Path, chunk_size: int = 512, overlap: int = 64
) -> list[DocumentChunk]:
    """Extract and chunk a document, retaining file hash and page provenance."""
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    page_texts, file_hash = load_document(path)
    chunks: list[DocumentChunk] = []
    for text, page_number in page_texts:
        for chunk_text in _split_text(text, chunk_size, overlap):
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=str(path.resolve()),
                    file_hash=file_hash,
                    chunk_index=len(chunks),
                    page_number=page_number,
                )
            )
    return chunks


def chunk_directory(
    directory: Path, chunk_size: int = 512, overlap: int = 64
) -> list[DocumentChunk]:
    """Chunk all supported documents beneath a directory in stable order."""
    extensions = {".pdf", ".docx", ".md", ".markdown"}
    paths = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    )
    return [
        chunk
        for path in paths
        for chunk in chunk_file(path, chunk_size=chunk_size, overlap=overlap)
    ]
