# Medium Article RAG Assistant

A Retrieval-Augmented Generation system that answers questions **only** from a corpus
of ~7,600 English Medium articles. It retrieves relevant passages from Pinecone and
answers from that context alone — no outside knowledge.

**Live API:** https://medium-rag-pi.vercel.app
&nbsp;•&nbsp; `GET /api/stats` &nbsp;•&nbsp; `POST /api/prompt` (`{"question": "..."}`)

## Layout

```
ingest/                 # one-time offline pipeline
  chunk.py              # CSV -> token chunks (derives stable article_id = row index)
  embed_and_upload.py   # chunk -> embed (cached) -> upsert to Pinecone
api/                    # Vercel serverless functions
  prompt.py             # POST /api/prompt
  stats.py              # GET  /api/stats
lib/
  config.py             # models, hyperparameters, env, system prompt (source of truth)
  chunking.py           # token-based chunking (tiktoken)
  rag.py                # shared retrieval + generation
data/                   # medium-english-50mb.csv (gitignored)
tests/
```

## Setup (local)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in the keys
```

`.env` (never committed):

```
LLMOD_API_KEY=...
LLMOD_BASE_URL=https://api.llmod.ai/v1
PINECONE_API_KEY=...
PINECONE_INDEX=medium-rag
```

## Ingestion (run once — mind the $5 budget)

Embeddings are cached locally (`ingest/.cache/embeddings.jsonl`), so re-runs never
re-embed unchanged chunks. **Validate on a small subset first, then scale.**

```powershell
# Preview chunking only (no API calls)
.\.venv\Scripts\python.exe -m ingest.chunk --limit 5

# Create the index + ingest ~20 articles to validate end-to-end
.\.venv\Scripts\python.exe -m ingest.embed_and_upload --limit 20 --create-index

# Full corpus
.\.venv\Scripts\python.exe -m ingest.embed_and_upload
```

## RAG hyperparameters

Set in `lib/config.py` and reported verbatim by `GET /api/stats`. Chosen via the
offline tuning harness — see **[REPORT.md](REPORT.md)** for method and results.

| param          | value | constraint        |
| -------------- | ----- | ----------------- |
| `chunk_size`   | 768   | int, max 1024     |
| `overlap_ratio`| 0.2   | float in [0, 0.3] |
| `top_k`        | 8     | int in [1, 30]    |

Re-run the comparison (local, cached, ~no cost) with:

```powershell
.\.venv\Scripts\python.exe -m pip install numpy   # dev-only, not a deploy dep
.\.venv\Scripts\python.exe -m ingest.experiment --subset 200
```

## API

`POST /api/prompt` — `{ "question": "..." }` → `{ response, context[], Augmented_prompt{System,User} }`
`GET /api/stats` — `{ chunk_size, overlap_ratio, top_k }`

## Deploy (Vercel)

Set the same four env vars as Vercel Environment Variables, then deploy. The `api/*.py`
files map automatically to `/api/prompt` and `/api/stats`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q        # if pytest installed
.\.venv\Scripts\python.exe tests/test_chunking.py     # or run standalone (no deps)
```
