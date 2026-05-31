# How this RAG system works (plain-English walkthrough)

This explains what we built and *why*, in reading order. It maps every concept to the
actual file that implements it. For the hyperparameter numbers and experiments, see
[REPORT.md](REPORT.md); for setup/commands, see [README.md](README.md).

---

## The one-sentence idea

**RAG = Retrieval-Augmented Generation.** Instead of letting the chat model answer from
its own training knowledge, we *retrieve* the most relevant Medium-article passages from
a search index and *paste them into the prompt*, then tell the model: "answer using only
this." That's how we guarantee answers come from the dataset and nothing else.

There are two completely separate halves:

1. **Ingestion (offline, run once):** turn 7,682 articles into a searchable vector index.
2. **Querying (online, per request):** look up passages for a question and have the model
   answer from them.

---

## Half 1 — Ingestion: building the searchable index

Goal: make every article *findable by meaning*, not just keywords.

### Step 1: Chunking — `lib/chunking.py` + `ingest/chunk.py`
Articles are long (hundreds–thousands of words). Models and search work better on small
pieces, so we split each article into overlapping **chunks**.

- We split by **tokens** (the unit models actually read), using `tiktoken`, so our size
  settings are exact. → `lib/chunking.py::chunk_text`
- Each chunk is **768 tokens** with **20% overlap** (so a sentence spanning a boundary
  isn't lost). These are the tuned values.
- The dataset has **no ID column**, so we assign each article a stable `article_id` = its
  row number, and carry it everywhere. → `ingest/chunk.py::iter_chunks`
- Result: ~7,682 articles → **18,914 chunks**.

### Step 2: Embedding — `ingest/embed_and_upload.py`
An **embedding** turns a piece of text into a list of 1,536 numbers (a "vector") that
captures its meaning. Texts about similar topics get similar vectors.

- We use the `text-embedding-3-small` model (via the LLMod proxy) to embed every chunk.
- **Cost control:** every embedding is cached to `ingest/.cache/embeddings.jsonl`, keyed
  by a hash of the text. Re-running never re-embeds the same chunk → we never pay twice.
  This is why the whole corpus cost ~$0.29 once and re-runs are free.

### Step 3: Upload to Pinecone — `ingest/embed_and_upload.py`
**Pinecone** is a vector database: a search engine for these meaning-vectors. We upload
all 18,914 vectors into an index called `medium-rag` (1536 dimensions, **cosine**
similarity). Alongside each vector we store **metadata** — `article_id`, `title`, `url`,
and the chunk text — so a search can return the answer *and* its source without a second
lookup.

> After this half, we have a cloud index that, given any question-vector, can instantly
> return the most semantically similar article passages.

---

## Half 2 — Querying: answering a question

This is what runs on every API call. All of it lives in `lib/rag.py::generate`.

### Step 1: Embed the question — `lib/rag.py::embed_query`
We embed the user's question with the *same* model, producing a 1,536-number vector in
the same "meaning space" as the chunks.

### Step 2: Retrieve — `lib/rag.py::retrieve`
We ask Pinecone for the **top_k = 8** chunks whose vectors are closest (highest cosine
similarity) to the question vector. Each comes back with its metadata and a similarity
`score`.

### Step 3: Deduplicate (for "list 3 articles") — `lib/rag.py::dedup_by_article`
A single article can produce several of the top chunks. For "give me 3 articles on X" we
collapse to the **highest-scoring chunk per `article_id`**, so we return 3 *distinct*
articles, not 3 pieces of the same one.

### Step 4: Build the augmented prompt — `lib/rag.py::build_user_prompt`
We assemble the retrieved passages into a numbered context block and append the question.
This is the "augmented" part of RAG — the prompt is *augmented* with retrieved evidence.

### Step 5: Generate the answer — `lib/rag.py::generate`
We call the chat model (`gpt-5-mini`) with two messages:
- **System prompt** (`lib/config.py::SYSTEM_PROMPT`): the strict instructions — *answer
  only from the provided context; if it's not there, say exactly "I don't know based on
  the provided Medium articles data."* We added a scoped clarification so the no-answer
  case returns that sentence **exactly**, with no extra explanation.
- **User prompt:** the augmented context + question from Step 4.

The model's reply is the final answer, grounded entirely in retrieved passages.

---

## The two API endpoints — `api/prompt.py`, `api/stats.py`

These are the deployed serverless functions (the public interface).

### `POST /api/prompt`
In: `{ "question": "..." }`. Out, exactly:
```json
{
  "response": "the model's grounded answer",
  "context": [{"article_id","title","chunk","score"}, ...],
  "Augmented_prompt": {"System": "...", "User": "..."}
}
```
`context` shows the evidence used; `Augmented_prompt` shows the exact prompt sent — full
transparency into *why* the model answered as it did.

### `GET /api/stats`
Returns the live hyperparameters: `{ "chunk_size": 768, "overlap_ratio": 0.2, "top_k": 8 }`.
Because these come straight from `lib/config.py`, the endpoint always reflects reality.

---

## The four query types it handles
1. **Precise fact** — find one specific article, return requested fields. (Verified: rank-1 hits.)
2. **List up to 3 articles** — retrieve, then dedup by `article_id` → distinct articles.
3. **Key-idea summary** — retrieve one relevant article, summarize from its chunks.
4. **Recommendation + justification** — recommend one article, justify from its text.

---

## Why the key numbers (short version)
- **chunk_size 768 / overlap 0.2 / top_k 8** were chosen by an offline experiment
  (`ingest/experiment.py`) comparing combos on an eval set across all four query types.
  768 produced the fewest chunks among the top scorers (cheaper) while giving the model
  fuller passages. Full reasoning + tables: [REPORT.md](REPORT.md).
- **Budget discipline:** embedding is the only real cost; caching + a single full ingest
  kept the whole project at ~$0.33 of the $5 budget.

---

## File map (where to look)
| Concern | File |
| --- | --- |
| Models, hyperparameters, system prompt | `lib/config.py` |
| Token chunking | `lib/chunking.py` |
| Retrieval + generation (the core) | `lib/rag.py` |
| Offline: chunk articles | `ingest/chunk.py` |
| Offline: embed + upload (cached) | `ingest/embed_and_upload.py` |
| Offline: hyperparameter tuning | `ingest/experiment.py` |
| API: answer a question | `api/prompt.py` |
| API: report config | `api/stats.py` |
| Tests (offline + live) | `tests/` |
