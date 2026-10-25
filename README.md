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

Requirements: Python 3.11+ and [Ollama](https://ollama.com/) for local answer
generation. No cloud LLM API key is needed. Local development uses Qdrant's
embedded persistent mode and does not require Docker. Docker Compose is an
optional alternative.

1. Create and activate a virtual environment, then install dependencies:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   Copy-Item .env.example .env
   ```

2. Install and start Ollama, then download the default model:

   ```powershell
   ollama pull llama3.2
   ollama list
   ```

   Ollama normally runs as a background service on Windows. If it is not
   running, start it with `ollama serve` in another terminal. Leave
   `LOCAL_MODEL_BASE_URL=http://localhost:11434/v1` and
   `LOCAL_MODEL_NAME=llama3.2` in `.env`. For the embedded local Qdrant database,
   leave `QDRANT_URL=local`. Start the API:

   ```powershell
   .venv\Scripts\python.exe -m uvicorn app.main:app --reload
   ```

   Local Qdrant data is persisted in `data\index\qdrant`. The API process owns
   this embedded database; do not start multiple API workers against the same
   local data path.

### Use the hosted Qdrant cluster locally

The local API can use the hosted Qdrant cluster while keeping its vectors
separate from production. In the ignored local `.env` file, set:

```dotenv
QDRANT_URL=https://your-qdrant-cluster-endpoint
QDRANT_API_KEY=your-local-qdrant-api-key
QDRANT_COLLECTION=documents_local
```

Replace the endpoint and key with your Qdrant cluster values; do not commit
`.env` or share the API key. The Render API uses the separate `documents`
collection. The Qdrant key must permit creating and reading/writing the local
collection. This repository's local `.env` uses `documents_local`; do not
change it to `documents` unless you intentionally want local ingestion and
production to share vectors. Restart the local API after changing these
settings, then ingest documents through the local UI. Qdrant vectors are stored
in the cluster, while the local BM25 corpus and semantic cache remain under
`DATA_DIR` on your computer. This setup does not automatically expose production
documents to local chat.

To use a hosted OpenAI-compatible language model locally, set `LLM_BASE_URL`,
`LLM_MODEL_NAME`, and `LLM_API_KEY` in `.env`. These take precedence over the
legacy `LOCAL_MODEL_*` variables, which remain supported for local Ollama.
Restart the API after changing model settings.

### Admin portal credentials

The UI opens in chat-only mode. To enable its **Admin** portal, configure these
secrets on both the local API and the Render API service:

- `ADMIN_USERNAME`: the administrator login name.
- `ADMIN_PASSWORD_HASH`: an Argon2id hash, not the plaintext password.
- `ADMIN_SESSION_SECRET`: a unique signing secret generated with
  `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

Generate the password hash interactively so the plaintext is not included in
the command or shell history:

```powershell
python -m src.admin.create_password_hash
```

Copy the printed hash into the ignored local `.env` as `ADMIN_PASSWORD_HASH`.
Set the same variable names in the Render API service's secret environment.
Never put the plaintext password, hash, or session signing secret in source
control. The UI sends credentials to the API over its server-side connection;
the API verifies the password with Argon2id and issues a signed admin session
that expires after 30 minutes. Five failed logins from the same client address
are temporarily throttled in each API process.

Only after successful login does the UI show document ingestion, chunk
metadata, Qdrant status, and safe service configuration. API keys are never
returned by the admin endpoints. Configure HTTPS for public deployments. The
login throttle is process-local; deployments with multiple API instances that
need coordinated lockout should use a shared rate-limiting store.

3. Upload the included handbook in the UI sidebar and click **Ingest documents**.
   Alternatively, start the optional Docker Compose stack instead of the local
   API:

   ```powershell
   docker compose up --build qdrant api
   ```

4. Start the web UI in another terminal:

   ```powershell
   .venv\Scripts\python.exe -m streamlit run app/ui.py
   ```

   Visit <http://localhost:8501> for the UI and <http://localhost:8000/docs> for
   interactive API documentation. For a Docker-only setup, run
   `docker compose up --build` to start Qdrant, API, and UI together.

Answer generation uses the local Ollama service. Set `LOCAL_MODEL_NAME` to a
model already pulled with Ollama, and set `LOCAL_MODEL_BASE_URL` if Ollama is
running at a different address. Embedding and reranking models are downloaded
from Hugging Face on first use, then run locally; set `EMBEDDING_MODEL` or
`RERANKER_MODEL` to choose other compatible models. For a remote Qdrant
instance, set `QDRANT_URL` and optionally `QDRANT_API_KEY`.

## API

- `GET /health` — liveness check.
- `GET /status` — authenticated Qdrant collection status and exact indexed chunk
  count. Requires the `X-API-Key` header when API-key authentication is enabled.
- `POST /admin/login` — verifies the admin username and Argon2id password hash
  and issues a signed 30-minute bearer token. It also requires the service
  `X-API-Key` when API-key authentication is enabled.
- `GET /admin/status`, `GET /admin/config`, and `GET /admin/chunks` — admin-only
  status and sanitized metadata endpoints. Require the signed admin bearer
  token. Configuration responses expose service hosts and whether keys are
  configured, never secret values.
- `POST /admin/ingest` — admin-token-protected document ingestion for the UI.
- `POST /ingest` — multipart upload; accepts `.pdf`, `.md`, `.markdown`, and
  `.docx` files, extracts/chunks them, and indexes them.
- `POST /chat` — accepts `{"query": "...", "top_k": 4, "history": [...]}`.
  `history` is optional and contains up to 8 prior `{ "role": "user" |
  "assistant", "content": "..." }` messages, with an 8,000-character total limit.
  The UI sends recent messages from the current Streamlit session so contextual
  follow-ups can be resolved. Social messages and acknowledgements are handled
  without retrieval or LLM calls; follow-ups without relevant history receive a
  clarification prompt. Context-dependent follow-ups bypass the semantic cache.
  The endpoint returns an answer, source/page/chunk citations, rerank scores,
  and whether the response was cached.

For example:

```powershell
curl.exe -X POST http://localhost:8000/ingest -F "files=@data\sample_docs\company_handbook.md"
curl.exe -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d "{\"query\":\"How many days of annual leave do employees receive?\"}"
```

For a contextual follow-up, include the preceding user and assistant turns:

```json
{
  "query": "Why?",
  "history": [
    {"role": "user", "content": "How much annual leave is provided?"},
    {"role": "assistant", "content": "The policy provides 20 days of annual leave."}
  ]
}
```

The local semantic cache uses normalized embedding cosine similarity with a
default threshold of `0.92`. Its SQLite database and the BM25 corpus are stored
under `DATA_DIR` (default `data/index`). Qdrant stores dense vectors. Chat
generation and evaluation use the configured Ollama endpoint; embeddings and
reranking execute locally after their model files are downloaded.

## Evaluation

The included dataset has 25 question/context/ground-truth examples based on the
sample handbook. First ingest the handbook, configure the LLM, then run:

```powershell
python -m src.evaluation.eval
```

Ragas produces `data/evaluation/report.json` and `report.md` with faithfulness,
answer relevancy, context precision, and context recall averages, using Ollama
for LLM-graded metrics and local embeddings for relevance. These are model- and
retrieval-dependent measurements, not precomputed benchmark claims.

## Development

```powershell
pytest
black --check src app tests
flake8 src app tests --max-line-length=88 --extend-ignore=E203,W503
pre-commit install
```

GitHub Actions runs tests and style checks on pushes and pull requests.
The **PR Gateway** workflow also builds the production Docker image. To block
merges when any gateway check fails, configure the repository's `main` branch
protection (or ruleset) to require the **PR Gateway / checks** status check.

## Deploying to Render

The `render.yaml` blueprint defines public API and UI web services. To deploy,
create a new Blueprint in Render from this repository and provide the prompted
environment values:

- `LOCAL_MODEL_BASE_URL`: the reachable OpenAI-compatible `/v1` endpoint for
  your hosted Ollama/model service. `localhost` points to the Render container,
  not your computer.
- `LOCAL_MODEL_NAME`: a model available at that endpoint.
- `LOCAL_MODEL_API_KEY`: the hosted model provider's API key; use `ollama` for a
  local Ollama-compatible endpoint that does not require a key.
- `QDRANT_URL` and `QDRANT_API_KEY`: an external Qdrant instance. Render's
  local filesystem and this app's embedded Qdrant mode are not suitable for
  persistent public deployment.
- `API_ACCESS_KEY`: create a long, random value and set the same secret on both
  the API and UI services. Render requires it for `/chat` and `/ingest`; the UI
  sends it as the `X-API-Key` header. Do not commit it or put it in a public
  frontend.
- `ADMIN_USERNAME`, `ADMIN_PASSWORD_HASH`, and `ADMIN_SESSION_SECRET`: configure
  these on the API service to enable protected admin login. Generate the Argon2id
  hash using `python -m src.admin.create_password_hash` and generate a separate
  signing secret with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
  Keep the plaintext password and signing secret private; never commit them.

After deployment, copy the API service's public URL from Render (for example,
`https://enterprise-rag-api.onrender.com`). Check its public health endpoint:

```powershell
curl.exe https://enterprise-rag-api.onrender.com/health
```

Use that URL instead of `http://localhost:8000` in external API clients. For
example, set the client base URL to the API URL and provide the key:

```powershell
$env:API_ACCESS_KEY = "your-generated-secret"
curl.exe -X POST https://enterprise-rag-api.onrender.com/chat `
  -H "X-API-Key: $env:API_ACCESS_KEY" `
  -H "Content-Type: application/json" `
  -d "{\"query\":\"Ask a question about your indexed documents\"}"
```

The blueprint connects the deployed UI to the API and configures both services
to use the same `API_ACCESS_KEY`; set the same value in each service's Render
environment. The API's `/health` endpoint remains public. Render's default
filesystem is ephemeral, so re-ingest documents after a service restart unless
you configure persistent storage for `DATA_DIR`; use external persistent
Qdrant for public deployment.

Set the `RENDER_DEPLOY_HOOK_URL` repository secret to enable the optional deploy
workflow. No public deployment is created automatically by this repository.

## Safety and production notes

The included prompt-injection check is a small pattern-based baseline, and the
answer grounding check uses lexical overlap. Neither is a substitute for a
dedicated moderation/classification service or a semantic entailment checker.
Use trusted documents, protect the API key, configure rate limiting before
exposing the API, and validate these safeguards against your threat model.
`/health` is a liveness check and does not assert that Qdrant or model services
are reachable.
