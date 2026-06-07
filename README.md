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

## Usage examples

Against the live deployment (swap in `http://localhost:3000` if running `vercel dev`).

Current hyperparameters:

```bash
curl https://medium-rag-pi.vercel.app/api/stats
# {"chunk_size": 768, "overlap_ratio": 0.2, "top_k": 8}
```

Ask a question grounded in the corpus:

```bash
curl -X POST https://medium-rag-pi.vercel.app/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "Which article is about becoming a better writer, and who wrote it?"}'
```

Response (context trimmed for brevity):

```json
{
  "response": "The article \"An Easy Way for Writers to Move From Skinny Ideas to Rock Solid First Drafts,\" written by Dawn Bevier, ...",
  "context": [
    {
      "article_id": "1234",
      "title": "An Easy Way for Writers to Move From Skinny Ideas...",
      "chunk": "the retrieved passage text...",
      "score": 0.61
    }
  ],
  "Augmented_prompt": {
    "System": "You are a Medium-article assistant that answers questions strictly and only based on...",
    "User": "Use only the following retrieved Medium article context to answer...\n\n=== CONTEXT ===\n..."
  }
}
```

A question the corpus cannot answer returns the fixed refusal string verbatim:

```bash
curl -X POST https://medium-rag-pi.vercel.app/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
# {"response": "I don't know based on the provided Medium articles data.", "context": [...], "Augmented_prompt": {...}}
```

On Windows PowerShell, `Invoke-RestMethod` is the native equivalent:

```powershell
Invoke-RestMethod -Uri https://medium-rag-pi.vercel.app/api/prompt -Method Post `
  -ContentType "application/json" `
  -Body '{"question": "Which article is about becoming a better writer, and who wrote it?"}'
```

## Deploy (Vercel)

Set the same four env vars as Vercel Environment Variables, then deploy. The `api/*.py`
files map automatically to `/api/prompt` and `/api/stats`.

## Tests

```powershell
.\.venv\Scripts\python.exe tests/test_chunking.py     # run standalone (no deps)
.\.venv\Scripts\python.exe tests/test_contract.py
```
