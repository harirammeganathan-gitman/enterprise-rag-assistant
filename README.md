# Enterprise RAG Assistant

A modular document-question-answering starter with PDF/Markdown/DOCX ingestion,
512-token recursive chunking, Qdrant dense retrieval, BM25, reciprocal-rank
fusion, cross-encoder reranking, a semantic response cache, a FastAPI API, and a
Streamlit interface.

## Architecture

```text
Documents -> parser/chunker -> Qdrant + local BM25 corpus
Question  -> input check -> semantic cache -> dense + BM25 -> RRF -> reranker
          -> context-only LLM answer -> grounding check -> cited response
```

`src/ingestion/` extracts text and preserves the source path, SHA-256 file hash,
chunk index, and PDF page number where available. Chunk lengths are estimated
with a word/punctuation tokenizer; the default is 512 estimated tokens with 64
tokens of overlap. The DOCX parser preserves text and tables, but DOCX page
numbers are not available from this extraction method.

## Quick start (local)

Requirements: Python 3.11+, Docker with Compose, and an OpenAI-compatible API
key for answer generation.

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

2. Add `OPENAI_API_KEY` to `.env`. Start Qdrant and the API:

   ```powershell
   docker compose up --build qdrant api
   ```

3. In another terminal, ingest the included demonstration handbook:

   ```powershell
   .venv\Scripts\python.exe -m src.ingestion.cli data\sample_docs
   ```

   The CLI prints chunk JSON for inspection. To index documents for chat, send
   them to `/ingest` (or use the Streamlit uploader); CLI output alone does not
   write to Qdrant.

4. Start the web UI:

   ```powershell
   docker compose up --build ui
   ```

   Visit <http://localhost:8501> for the UI and <http://localhost:8000/docs> for
   interactive API documentation.

The model and first embedding pass download model weights from Hugging Face.
Set `EMBEDDING_MODEL`, `RERANKER_MODEL`, and `OPENAI_MODEL` in `.env` to use
different models. For a remote Qdrant instance, set `QDRANT_URL` and optionally
`QDRANT_API_KEY`.

## API

- `GET /health` — liveness check.
- `POST /ingest` — multipart upload; accepts `.pdf`, `.md`, `.markdown`, and
  `.docx` files, extracts/chunks them, and indexes them.
- `POST /chat` — JSON body `{"query": "...", "top_k": 4}`; returns an answer,
  source/page/chunk citations, rerank scores, and whether the response was cached.

For example:

```powershell
curl.exe -X POST http://localhost:8000/ingest -F "files=@data\sample_docs\company_handbook.md"
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"How many days of annual leave do employees receive?\"}"
```

The local semantic cache uses normalized embedding cosine similarity with a
default threshold of `0.92`. Its SQLite database and the BM25 corpus are stored
under `DATA_DIR` (default `data/index`). Qdrant stores dense vectors.
When both `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are configured, the
Langfuse OpenAI wrapper records generation traces and token usage; without those
keys the application does not export traces. Enabling Langfuse sends prompts
and retrieved context to that configured service.

## Evaluation

The included dataset has 25 question/context/ground-truth examples based on the
sample handbook. First ingest the handbook, configure the LLM, then run:

```powershell
python -m src.evaluation.eval
```

Ragas produces `data/evaluation/report.json` and `report.md` with faithfulness,
answer relevancy, context precision, and context recall averages. These are
model- and retrieval-dependent measurements, not precomputed benchmark claims.

## Development

```powershell
pytest
black --check src app tests
flake8 src app tests --max-line-length=88 --extend-ignore=E203,W503
pre-commit install
```

GitHub Actions runs tests and style checks on pushes and pull requests. The
Render blueprint (`render.yaml`) defines API and UI web services; add the
required model/API keys and external Qdrant URL in Render. Set the
`RENDER_DEPLOY_HOOK_URL` repository secret to enable the optional deploy
workflow. No public deployment is created automatically by this repository.
Render's default filesystem is ephemeral, so re-ingest documents after a service
restart unless you configure persistent storage for `DATA_DIR`; Qdrant should be
configured as an external persistent service.

## Safety and production notes

The included prompt-injection check is a small pattern-based baseline, and the
answer grounding check uses lexical overlap. Neither is a substitute for a
dedicated moderation/classification service or a semantic entailment checker.
Use trusted documents, configure authentication and rate limiting before
exposing the API, and validate these safeguards against your threat model.
`/health` is a liveness check and does not assert that Qdrant or model services
are reachable.
