# Multimodal Financial RAG

A production-grade **Retrieval-Augmented Generation** system that turns financial PDFs into instant, cited answers — with full chart and figure understanding.

Upload an annual report, ask a question, get a precise answer with source citations in seconds — grounded entirely in the document, no hallucinations.

**Example:**
> "What was Infosys operating margin in FY2025 vs FY2024?"
> → "Infosys operating margin improved from **20.7% in FY2024** to **21.1% in FY2025**. [Source 2]"

---

## How It Works (Two Phases)

**Ingest Phase (run once per document):**
1. Extract text blocks and embedded images from PDF (PyMuPDF)
2. Detect charts using CLIP zero-shot classification
3. Caption charts with Amazon Nova / Claude Vision via Bedrock
4. Chunk text into overlapping nodes merged by page (~800 nodes per 369-page report)
5. Embed nodes locally with `all-MiniLM-L6-v2` (384-dim, no API cost, ~3 seconds for 800 chunks)
6. Persist vector index to disk

**Query Phase (every question):**
1. **BM25 Search** — keyword matching
2. **Vector Search** — semantic similarity (local sentence-transformers)
3. **RRF Fusion** — combines both ranked lists
4. **Cross-Encoder Reranking** — picks the most relevant chunks
5. **Amazon Nova Lite** — generates grounded answer with source citations
6. If charts are on the same page as retrieved chunks → sent as images to Bedrock Vision

---

## Key Features

✅ **Multimodal** — Understands charts, graphs, and figures (CLIP + Bedrock Vision)  
✅ **Hybrid Search** — BM25 + dense vector + Reciprocal Rank Fusion  
✅ **Cross-Encoder Reranking** — Only top chunks sent to LLM  
✅ **Grounded Answers** — No hallucinations, cited directly from document  
✅ **Fast Indexing** — 369-page PDF indexed in under 1 minute  
✅ **Async Ingestion** — Background job with polling endpoint  
✅ **Local Embeddings** — No embedding API cost (sentence-transformers)  
✅ **AWS Native** — Bedrock LLM, S3, Lambda, DynamoDB, K8s ready  
✅ **Financial NER** — LoRA fine-tuned BERT for ORG / MONEY / DATE / PERCENT  
✅ **Built-in UI** — Upload PDFs and query from browser at `http://localhost:8000`  

---

## Project Structure

```
multimodal-finrag/
├── src/
│   ├── config.py                  # Pydantic settings (all env vars)
│   ├── ingestion/
│   │   ├── pdf_parser.py          # PyMuPDF text + image extraction
│   │   ├── chart_extractor.py     # CLIP chart detection + Bedrock captioning
│   │   └── s3_loader.py           # S3 upload/download/presigned URLs
│   ├── rag/
│   │   ├── bedrock_llm.py         # LlamaIndex CustomLLM for Bedrock (Claude 3 + Nova)
│   │   ├── embeddings.py          # Local sentence-transformers embedding (no API)
│   │   ├── retriever.py           # Hybrid BM25 + vector + cross-encoder reranker
│   │   └── pipeline.py            # End-to-end RAG pipeline
│   ├── finetune/
│   │   ├── dataset.py             # Financial NER dataset + synthetic data generator
│   │   ├── lora_trainer.py        # LoRA fine-tuning with PEFT + seqeval metrics
│   │   └── inference.py           # Entity extraction inference engine
│   ├── lambda_handler/
│   │   ├── handler.py             # Lambda S3 event handler
│   │   └── Dockerfile             # Container image for Lambda
│   └── api/
│       ├── main.py                # FastAPI app with lifespan
│       ├── schemas.py             # Pydantic v2 request/response models
│       └── routes/
│           ├── ingest.py          # POST /ingest (async background job)
│           ├── query.py           # POST /query
│           └── entities.py        # POST /entities (NER)
├── k8s/                           # Kubernetes manifests (HPA, Ingress, PVC)
├── scripts/                       # CLI tools (build index, deploy lambda, train LoRA)
├── artifacts/                     # Q&A docs, evaluation reports
└── tests/                         # pytest test suite
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI + Uvicorn |
| **RAG Orchestration** | LlamaIndex 0.10+ |
| **LLM** | Amazon Nova Lite / Claude 3 (AWS Bedrock) |
| **Embeddings** | `all-MiniLM-L6-v2` (local, sentence-transformers) |
| **Chart Detection** | OpenCLIP (`ViT-B-32`) zero-shot classification |
| **Chart Captioning** | Amazon Nova / Claude 3 Vision (Bedrock) |
| **Vector Search** | LlamaIndex VectorStoreIndex (disk-persisted) |
| **Keyword Search** | BM25 (rank-bm25) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **PDF Parsing** | PyMuPDF (fitz) |
| **NER Fine-tuning** | LoRA + PEFT on `bert-base-cased` |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker |
| **Orchestration** | Kubernetes (HPA + Ingress) |
| **Cloud** | AWS (Bedrock, S3, Lambda, DynamoDB) |

---

## Quick Start (Local)

### 1. Clone and install

```bash
git clone https://github.com/pritmon/multimodal-finrag.git
cd multimodal-finrag

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — add your AWS credentials and region
```

Minimum required in `.env`:
```
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

### 3. Start the API server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser — the built-in UI lets you upload PDFs and run queries directly.

### 4. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
     -F "file=@annual_report.pdf"
# Returns: {"job_id": "abc123", "status": "processing"}
```

Poll for completion:
```bash
curl http://localhost:8000/ingest/status/abc123
# Returns: {"status": "done", "text_nodes": 905, "chart_nodes": 5}
```

### 5. Query the document

```bash
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What was the operating margin in FY2025 vs FY2024?"}'
```

### 6. Extract financial entities

```bash
curl -X POST http://localhost:8000/entities \
     -H "Content-Type: application/json" \
     -d '{"text": "Infosys reported ₹1,62,990 crore revenue in FY2025 with 21.1% operating margin."}'
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve built-in UI |
| `/health` | GET | Health check + index status |
| `/ingest` | POST | Upload PDF (async background job) |
| `/ingest/status/{job_id}` | GET | Poll job progress |
| `/query` | POST | RAG query with source citations |
| `/entities` | POST | Financial NER extraction |
| `/docs` | GET | Swagger API documentation |

**Request body for `/query`:**
```json
{
  "question": "Your question here",
  "top_k": 8
}
```

**Response:**
```json
{
  "answer": "Infosys operating margin improved from 20.7% (FY2024) to 21.1% (FY2025). [Source 2]",
  "sources": [
    { "text": "...chunk text...", "score": 0.91, "metadata": { "page_number": 42 } }
  ],
  "charts": [
    { "caption": "Bar chart showing segment revenue...", "page_number": 15, "image_b64": "..." }
  ]
}
```

---

## AWS Setup

### Required IAM permissions

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.nova-lite-v1:0",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-*",
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": "*"
    }
  ]
}
```

### Enable Bedrock models

In the AWS Console → Bedrock → Model Access, enable:
- `Amazon Nova Lite` (recommended — used for queries and chart captioning)
- `Anthropic Claude 3 Sonnet` (alternative)

---

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -k k8s/

# Verify
kubectl -n finrag get pods
kubectl -n finrag get hpa

# Logs
kubectl -n finrag logs -l app=finrag-api --tail=100
```

---

## NER Label Schema

| Label | Description | Example |
|-------|-------------|---------|
| `B-ORG` / `I-ORG` | Organisation name | *Infosys*, *Tesla* |
| `B-MONEY` / `I-MONEY` | Monetary amount | *₹1,62,990 crore*, *$22.4 billion* |
| `B-DATE` / `I-DATE` | Date or fiscal period | *Q1 2026*, *FY2025* |
| `B-PERCENT` / `I-PERCENT` | Percentage figure | *21.1%*, *40 basis points* |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `AWS_ACCESS_KEY_ID` | — | AWS access key |
| `AWS_SECRET_ACCESS_KEY` | — | AWS secret key |
| `S3_BUCKET` | `finrag-documents` | Document storage bucket |
| `BEDROCK_MODEL_ID` | `amazon.nova-lite-v1:0` | Generation model |
| `INDEX_PERSIST_DIR` | `./index_store` | Vector index storage path |
| `LORA_MODEL_PATH` | `./models/finrag-ner-lora` | NER adapter path |
| `RETRIEVER_TOP_K` | `8` | Documents retrieved per query |
| `RERANKER_TOP_N` | `4` | Documents after reranking |
| `CHUNK_SIZE` | `512` | Token chunk size |
| `CHUNK_OVERLAP` | `64` | Token overlap between chunks |

---

## Performance

| Metric | Value |
|--------|-------|
| Indexing speed (369-page PDF) | ~41 seconds |
| Embedding speed (800 chunks) | ~3 seconds (local, no API cost) |
| Charts detected & captioned | Up to 5 per document (parallel) |
| Query latency | ~4 seconds |

---

## Running Tests

```bash
# All tests
pytest

# Fast tests only
pytest -m "not slow"

# With coverage
pytest --cov=src --cov-report=html
```

---

Built by **Pritam Mondal** — MIT License
