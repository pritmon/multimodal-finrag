# multimodal-finrag — Project Q&A

Interview-style questions and answers about the system architecture, design decisions, and implementation.

---

## 1. What is this project?

A **Multimodal Financial Document Intelligence** system that lets you upload PDF financial documents (annual reports, earnings updates) and ask natural language questions about them.

It combines:
- **RAG** (Retrieval-Augmented Generation) for accurate, cited answers
- **Multimodal** understanding (text + charts/images)
- **AWS Bedrock** for LLM inference and embeddings
- **FastAPI** backend with async document indexing
- **LlamaIndex** for vector indexing and retrieval

---

## 2. What is RAG and why use it here?

**RAG = Retrieval-Augmented Generation.**

Instead of relying on an LLM's training data (which may be outdated or hallucinated), RAG:
1. Retrieves the most relevant chunks from your documents
2. Passes them as context to the LLM
3. LLM answers based only on retrieved content

**Why it matters for finance:** Revenue figures, EPS, margins change every quarter. A static LLM would give wrong numbers. RAG pulls the exact figures from the uploaded document.

---

## 3. Walk me through the ingestion pipeline.

```
PDF Upload → S3 → Background Thread
               ↓
         PyMuPDF (fitz)
         - Extract text blocks per page
         - Merge blocks by page → one Document per page
               ↓
         SentenceSplitter (512 tokens, 64 overlap)
         - 369 pages → ~800 nodes
               ↓
         BedrockTitanEmbedding (parallel, 5 workers)
         - Each node → 1536-dim vector
               ↓
         VectorStoreIndex (LlamaIndex)
         - Persisted to disk (index_store/)
```

**Key optimization:** Originally created one Document per text block (11,327 nodes). Merged to one per page → 800 nodes → 10x faster indexing.

---

## 4. What is the retrieval strategy?

**Hybrid retrieval = BM25 + Vector + RRF + Cross-encoder reranking**

| Step | Method | Purpose |
|------|--------|---------|
| 1 | BM25 (keyword) | Exact term matching — great for numbers, tickers |
| 2 | Vector (semantic) | Meaning-based — handles paraphrasing |
| 3 | RRF fusion | Merge both ranked lists into one |
| 4 | Cross-encoder reranking | Re-score top candidates with query-aware model |

**Why hybrid?** Pure vector search misses exact numbers ("22,387"). Pure BM25 misses semantic queries ("how profitable is Tesla"). Combining both gives best recall.

---

## 5. What is RRF (Reciprocal Rank Fusion)?

A score merging formula:

```
RRF_score(doc) = Σ 1 / (k + rank)
```

Where `k=60` is a constant that dampens rank differences. A document ranked #1 in both BM25 and vector gets the highest fused score. It's parameter-light and works well in practice.

---

## 6. What is the cross-encoder reranker and why use it?

**Cross-encoder:** A transformer that takes `(query, passage)` as joint input and outputs a relevance score. More accurate than dot-product similarity but too slow to run on all documents.

**Strategy:** Run cheap retrieval (BM25 + vector) to get top-20 candidates, then run expensive cross-encoder only on those 20. Get accuracy of cross-encoder at the cost of ~20 inference calls.

**Model used:** `cross-encoder/ms-marco-MiniLM-L-6-v2` — small, fast, trained on MS MARCO passage ranking.

**Scores:** Raw logits normalized via sigmoid → clean 0–100% relevance scores shown in UI.

---

## 7. What LLM and embeddings are used?

| Component | Model | Provider |
|-----------|-------|----------|
| LLM | `amazon.nova-lite-v1:0` | AWS Bedrock |
| Embeddings | `amazon.titan-embed-text-v1` | AWS Bedrock |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace (local) |

**Why Nova Lite?** All Anthropic Claude models require payment instrument verification on AWS. Nova Lite works immediately on the free tier.

**Nova vs Claude API differences:**
- Nova request: `{"messages": [...], "inferenceConfig": {"max_new_tokens": N}}`
- Nova content: `[{"text": "..."}]` (no `"type"` field)
- Nova response: `result["output"]["message"]["content"][0]["text"]`

---

## 8. How is async indexing implemented?

```python
# Returns instantly after S3 upload
asyncio.get_event_loop().run_in_executor(
    None, _index_document, job_id, content, filename, settings
)

# Background thread updates job status
_jobs[job_id] = {"status": "indexing", ...}
# ... after done:
_jobs[job_id] = {"status": "done", "text_nodes": 914, ...}

# UI polls every 5s
GET /ingest/status/{job_id}
```

The upload returns in ~2s (S3 upload time). Indexing runs in a thread pool. UI polls until status is `"done"`.

---

## 9. What was the biggest performance challenge and how did you fix it?

**Problem:** Indexing a 369-page PDF took 10+ minutes.

**Root causes (found one by one):**
1. `insert_nodes([node])` called one at a time → LlamaIndex called `_get_text_embedding` per node (sequential, 1.37s each × 11,327 nodes = 4+ hours)
2. 11,327 nodes created (one per text block instead of per page)
3. Image extraction rasterizing all 369 pages even when unused
4. 20 parallel Bedrock calls → ThrottlingException

**Fixes applied:**
1. Merge text blocks by page → 800 nodes
2. Pre-embed all nodes with parallel `ThreadPoolExecutor` (5 workers, respects rate limit)
3. Skip image extraction
4. Insert all nodes in one `insert_nodes(nodes)` call

**Result:** ~4 minutes for 369 pages (rate-limited by AWS Bedrock free tier quota).

---

## 10. What is LoRA finetuning used for?

**LoRA (Low-Rank Adaptation)** finetunes BERT for financial Named Entity Recognition (NER).

Labels: `B-ORG`, `I-ORG`, `B-MONEY`, `I-MONEY`, `B-DATE`, `I-DATE`, `B-PERCENT`, `I-PERCENT`, `O`

**Why LoRA?** Full BERT finetuning updates 110M parameters. LoRA freezes the base model and adds small low-rank matrices (rank=8) to attention layers — trains ~1% of parameters with similar accuracy.

**Status:** Model not yet trained (`models/finrag-ner-lora` not present). NER endpoint returns 503 until trained.

---

## 11. What does the Entities tab in the UI do?

Sends text to `POST /entities` which runs the LoRA-finetuned BERT NER model. Returns tagged financial entities:

```json
[
  {"text": "Infosys", "label": "ORG", "start": 0, "end": 7},
  {"text": "₹1,36,592 crore", "label": "MONEY", "start": 20, "end": 36}
]
```

Currently unavailable until the LoRA model is trained.

---

## 12. How is AWS integrated?

| Service | Usage |
|---------|-------|
| S3 | Store uploaded PDFs (`pritam-finrag-docs` bucket) |
| Bedrock | LLM inference (Nova Lite) + embeddings (Titan) |
| Lambda | Async indexing handler (optional deployment) |

Credentials passed via environment variables, forwarded through `boto3.Session(**session_kwargs)` to all clients.

---

## 13. What are the API endpoints?

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status + index loaded flag |
| POST | `/ingest` | Upload PDF → returns job_id instantly |
| GET | `/ingest/status/{job_id}` | Poll indexing progress |
| POST | `/query` | Ask a question, get answer + sources |
| POST | `/entities` | Extract financial NER entities |

---

## 14. What would you improve with more time?

1. **Re-enable chart captioning** — OpenCLIP chart detection + Nova vision captioning for true multimodal Q&A
2. **Train the LoRA NER model** — needs annotated financial text dataset
3. **Increase Bedrock quota** — request higher TPS for faster indexing
4. **Streaming responses** — stream LLM tokens to UI for faster perceived latency
5. **Multi-document queries** — currently all documents share one index; add per-document filtering
6. **Persistent job store** — `_jobs` dict is in-memory, lost on restart; use Redis or DB
7. **Authentication** — no auth currently; add API keys or OAuth for production
