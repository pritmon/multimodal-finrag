<div align="center">

# 🧠 Multimodal Financial RAG

### Turn any financial PDF into instant, cited answers — with chart and image understanding.

[![Python CI](https://img.shields.io/github/actions/workflow/status/pritmon/multimodal-finrag/ci.yml?label=Python%20CI&logo=github)](https://github.com/pritmon/multimodal-finrag)
[![Live on EKS](https://img.shields.io/badge/Live-AWS%20EKS-brightgreen?logo=amazonaws)](http://finrag.44.206.217.242.nip.io)
[![Live on ECS](https://img.shields.io/badge/Live-AWS%20ECS-blue?logo=amazonaws)](http://13.222.137.204:8000)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![AWS Bedrock](https://img.shields.io/badge/AWS-Bedrock-orange?logo=amazonaws&logoColor=white)](https://aws.amazon.com/bedrock/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## What is this?

Financial analysts spend **days** reading dense annual reports, 10-Ks, and earnings releases.

**Multimodal Financial RAG** reads those documents for you — including the charts and figures. Upload a PDF, ask a question, and get a precise answer with the exact source it came from — in under 5 seconds.

> *"What was Infosys operating margin in FY2025 vs FY2024?"*
> → **"Operating margin improved from 20.7% (FY2024) to 21.1% (FY2025). [Source 2, Page 42]"**

It doesn't guess. It only answers from what's in the document — text **and** images.

---

## What makes it multimodal?

Most RAG systems only read text. This one reads **both**:

| Input Type | How it's handled |
|------------|-----------------|
| 📄 **Text** | PyMuPDF extracts paragraphs, tables, numbers — chunked and indexed |
| 🖼️ **Images** | CLIP detects charts/figures → Bedrock Vision captions them → included in search and answer |

So when you ask *"what was the revenue trend?"*, it finds the bar chart on that page, sends it to the LLM as an image, and explains it alongside the text.

---

## How It Works

**Ingest Phase** (run once per document):

```
PDF → PyMuPDF → Text Blocks + Images
                     │                └── CLIP Chart Detection
                     │                         └── Bedrock Vision Captioning
                     ▼
              Page-level Chunks (~800 nodes per 369-page report)
                     ▼
         all-MiniLM-L6-v2 Embeddings (local, ~3 seconds)
                     ▼
              Vector Index (disk-persisted)
```

**Query Phase** (every question):

```
Question → BM25 Search ──┐
         → Vector Search ─┤ RRF Fusion → Cross-Encoder Rerank → Nova Lite LLM → Answer + Citations
                          │                                          ↑
                    Chart images on same page ──────────────────────┘
```

---

## Key Features

✅ **Multimodal** — Reads text, images, charts, and figures from PDFs  
✅ **Hybrid Search** — BM25 + dense vector + Reciprocal Rank Fusion  
✅ **Cross-Encoder Reranking** — Only the most relevant chunks sent to LLM  
✅ **Grounded Answers** — No hallucinations, every answer cited from the document  
✅ **Fast Indexing** — 369-page PDF indexed in under 1 minute  
✅ **Async Ingestion** — Background job with real-time status polling  
✅ **Free Embeddings** — Local `all-MiniLM-L6-v2`, no API cost  
✅ **AWS Native** — Bedrock LLM + Vision, S3, Lambda, DynamoDB, K8s ready  
✅ **Financial NER** — LoRA fine-tuned BERT for ORG / MONEY / DATE / PERCENT  
✅ **Built-in UI** — Upload PDFs and query from browser at `http://localhost:8000`  

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
# Edit .env — add your AWS credentials
```

Minimum required:
```env
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
```

### 3. Start the server

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — built-in UI to upload PDFs and run queries.

### 4. Ingest a document

```bash
curl -X POST http://localhost:8000/ingest \
     -F "file=@annual_report.pdf"
# → {"job_id": "abc123", "status": "processing"}

# Poll for completion
curl http://localhost:8000/ingest/status/abc123
# → {"status": "done", "text_nodes": 905, "chart_nodes": 5}
```

### 5. Query

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
| `/` | GET | Built-in UI |
| `/health` | GET | Health check + index status |
| `/ingest` | POST | Upload PDF (async background job) |
| `/ingest/status/{job_id}` | GET | Poll job progress |
| `/query` | POST | RAG query with source citations |
| `/entities` | POST | Financial NER extraction |
| `/docs` | GET | Swagger API docs |

**Response from `/query`:**
```json
{
  "answer": "Operating margin improved from 20.7% (FY2024) to 21.1% (FY2025). [Source 2]",
  "sources": [
    { "text": "...chunk...", "score": 0.91, "metadata": { "page_number": 42 } }
  ],
  "charts": [
    { "caption": "Bar chart showing segment revenue by geography...", "page_number": 15 }
  ]
}
```

---

## Project Structure

```
multimodal-finrag/
├── src/
│   ├── config.py                  # Pydantic settings
│   ├── ingestion/
│   │   ├── pdf_parser.py          # PyMuPDF text + image extraction
│   │   ├── chart_extractor.py     # CLIP detection + Bedrock captioning
│   │   └── s3_loader.py           # S3 upload/download
│   ├── rag/
│   │   ├── bedrock_llm.py         # LlamaIndex CustomLLM (Claude 3 + Nova)
│   │   ├── embeddings.py          # Local sentence-transformers embedding
│   │   ├── retriever.py           # Hybrid BM25 + vector + reranker
│   │   └── pipeline.py            # End-to-end RAG pipeline
│   ├── finetune/
│   │   ├── dataset.py             # Financial NER dataset + synthetic data
│   │   ├── lora_trainer.py        # LoRA fine-tuning with PEFT
│   │   └── inference.py           # NER inference engine
│   ├── lambda_handler/
│   │   ├── handler.py             # Lambda S3 event handler
│   │   └── Dockerfile
│   └── api/
│       ├── main.py                # FastAPI app
│       ├── schemas.py             # Pydantic v2 models
│       └── routes/
│           ├── ingest.py          # POST /ingest
│           ├── query.py           # POST /query
│           └── entities.py        # POST /entities
├── k8s/                           # Kubernetes manifests
├── scripts/                       # CLI tools
├── artifacts/                     # Q&A docs, evaluation reports
└── tests/                         # pytest suite
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
      "Action": ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
      "Resource": [
        "arn:aws:bedrock:*::foundation-model/amazon.nova-lite-v1:0",
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-*"
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

Enable in AWS Console → Bedrock → Model Access: **Amazon Nova Lite** (or Claude 3 Sonnet)

---

## Performance

| Metric | Value |
|--------|-------|
| Indexing — 369-page PDF | ~41 seconds |
| Embedding — 800 chunks | ~3 seconds (local, free) |
| Charts detected per document | up to 5 (parallel captioning) |
| Query latency | ~4 seconds |

---

## NER Label Schema

| Label | Description | Example |
|-------|-------------|---------|
| `B-ORG` / `I-ORG` | Organisation | *Infosys*, *Tesla* |
| `B-MONEY` / `I-MONEY` | Monetary amount | *₹1,62,990 crore*, *$22.4B* |
| `B-DATE` / `I-DATE` | Date / fiscal period | *Q1 2026*, *FY2025* |
| `B-PERCENT` / `I-PERCENT` | Percentage | *21.1%*, *40 bps* |

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `BEDROCK_MODEL_ID` | `amazon.nova-lite-v1:0` | Generation model |
| `INDEX_PERSIST_DIR` | `./index_store` | Vector index path |
| `LORA_MODEL_PATH` | `./models/finrag-ner-lora` | NER adapter path |
| `RETRIEVER_TOP_K` | `8` | Chunks retrieved per query |
| `RERANKER_TOP_N` | `4` | Chunks after reranking |
| `CHUNK_SIZE` | `512` | Token chunk size |
| `CHUNK_OVERLAP` | `64` | Token overlap |

---

## Running Tests

```bash
pytest                        # all tests
pytest -m "not slow"          # skip model-loading tests
pytest --cov=src              # with coverage
```

---

<div align="center">

## Live Deployments

| Platform | URL |
|----------|-----|
| **AWS EKS** (Kubernetes) | http://finrag.44.206.217.242.nip.io |
| **AWS ECS** (Fargate) | http://13.222.137.204:8000 |

---

Built by **Pritam Mondal** — MIT License

</div>
