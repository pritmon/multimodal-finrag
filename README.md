# Multimodal Financial Document Intelligence (FinRAG)

A production-grade **Retrieval-Augmented Generation** system for financial documents, combining:

- **Multimodal PDF ingestion** — text extraction (PyMuPDF) + chart detection (CLIP) + captioning (Bedrock Claude 3 Vision)
- **Hybrid RAG pipeline** — BM25 + dense vector retrieval with cross-encoder reranking (LlamaIndex + Bedrock Titan Embeddings)
- **Financial NER fine-tuning** — LoRA-adapted BERT for ORG / MONEY / DATE / PERCENT entity extraction (PEFT)
- **AWS-native architecture** — S3 document store, Bedrock LLMs, Lambda async processor, DynamoDB metadata
- **FastAPI REST API** — `/ingest`, `/query`, `/entities` endpoints
- **Kubernetes deployment** — HPA, Ingress (nginx + TLS), PVC for shared index storage

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI (K8s)                        │
│  POST /ingest  │  POST /query   │  POST /entities           │
└──────┬─────────┴──────┬─────────┴──────────┬────────────────┘
       │                │                    │
       ▼                ▼                    ▼
  S3 Upload       FinRAGPipeline       NERInferenceEngine
  + Lambda        (LlamaIndex)         (PEFT LoRA BERT)
  trigger         │
                  ├─ BM25Retriever
                  ├─ VectorStoreIndex  ← Bedrock Titan Embeddings
                  ├─ CrossEncoder reranker
                  └─ Bedrock Claude 3 generation
                        │
                  ┌─────┴──────┐
               PDFParser   ChartExtractor
               (PyMuPDF)   (CLIP + Claude Vision)
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo>
cd multimodal-finrag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your AWS credentials, S3 bucket, etc.
```

### 3. Build the index

```bash
# From local PDFs
python scripts/build_index.py --local-dir /path/to/pdfs --output-dir ./index_store

# From S3
python scripts/build_index.py --s3-prefix documents/2024/ --output-dir ./index_store
```

### 4. Train the NER model (optional)

```bash
# Quick start with synthetic data
python scripts/train_lora.py --synthetic --output-dir ./models/finrag-ner-lora

# With real annotated data
python scripts/train_lora.py \
    --data-path data/financial_ner.jsonl \
    --output-dir ./models/finrag-ner-lora \
    --epochs 10 --lr 2e-4
```

### 5. Run the API

```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### 6. Query the API

```bash
# Health check
curl http://localhost:8000/health

# Ingest a document
curl -X POST http://localhost:8000/ingest \
     -F "file=@annual_report.pdf"

# RAG query
curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What was Goldman Sachs revenue in Q4 2023?", "top_k": 8}'

# Entity extraction
curl -X POST http://localhost:8000/entities \
     -H "Content-Type: application/json" \
     -d '{"text": "Goldman Sachs reported $47.3 billion in revenue for fiscal year 2023."}'
```

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
│   │   ├── bedrock_llm.py         # LlamaIndex CustomLLM for Bedrock Claude 3
│   │   ├── embeddings.py          # LlamaIndex BaseEmbedding for Bedrock Titan
│   │   ├── retriever.py           # Hybrid BM25 + vector + cross-encoder reranker
│   │   └── pipeline.py            # End-to-end RAG pipeline
│   ├── finetune/
│   │   ├── dataset.py             # Financial NER HuggingFace dataset + synthetic data
│   │   ├── lora_trainer.py        # LoRA fine-tuning with PEFT + seqeval metrics
│   │   └── inference.py           # Entity extraction inference engine
│   ├── lambda_handler/
│   │   ├── handler.py             # Lambda S3 event handler
│   │   └── Dockerfile             # Container image for Lambda
│   └── api/
│       ├── main.py                # FastAPI app with lifespan
│       ├── schemas.py             # Pydantic v2 request/response models
│       └── routes/
│           ├── ingest.py          # POST /ingest
│           ├── query.py           # POST /query
│           └── entities.py        # POST /entities
├── k8s/                           # Kubernetes manifests
├── scripts/                       # CLI tools
└── tests/                         # pytest test suite
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
        "arn:aws:bedrock:*::foundation-model/anthropic.claude-3-*",
        "arn:aws:bedrock:*::foundation-model/amazon.titan-embed-*"
      ]
    },
    { "Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], "Resource": "*" },
    { "Effect": "Allow", "Action": ["lambda:InvokeFunction"], "Resource": "*" },
    { "Effect": "Allow", "Action": ["dynamodb:PutItem", "dynamodb:GetItem"], "Resource": "*" }
  ]
}
```

### DynamoDB table

```bash
aws dynamodb create-table \
    --table-name finrag-document-metadata \
    --attribute-definitions AttributeName=document_id,AttributeType=S \
    --key-schema AttributeName=document_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST
```

---

## Kubernetes Deployment

```bash
# Apply all manifests
kubectl apply -k k8s/

# Verify deployment
kubectl -n finrag get pods
kubectl -n finrag get hpa

# Check logs
kubectl -n finrag logs -l app=finrag-api --tail=100
```

---

## Lambda Deployment

```bash
# Container image (recommended)
python scripts/deploy_lambda.py container \
    --ecr-repo finrag-lambda \
    --function-name finrag-document-processor \
    --region us-east-1

# ZIP package
python scripts/deploy_lambda.py zip \
    --function-name finrag-document-processor \
    --s3-bucket my-lambda-packages
```

---

## Running Tests

```bash
# All tests
pytest

# Fast tests only (no slow model loading)
pytest -m "not slow"

# With coverage
pytest --cov=src --cov-report=html
```

---

## NER Label Schema

| Label | Description | Example |
|-------|-------------|---------|
| `B-ORG` / `I-ORG` | Organisation name | *Goldman Sachs*, *Federal Reserve* |
| `B-MONEY` / `I-MONEY` | Monetary amount | *$47.3 billion*, *€1.2 trillion* |
| `B-DATE` / `I-DATE` | Date or fiscal period | *Q4 2023*, *fiscal year 2024* |
| `B-PERCENT` / `I-PERCENT` | Percentage figure | *15%*, *25 basis points* |

---

## Configuration Reference

All settings are loaded from environment variables or `.env`. See `.env.example` for the full list.

| Variable | Default | Description |
|----------|---------|-------------|
| `AWS_REGION` | `us-east-1` | AWS region |
| `S3_BUCKET` | `finrag-documents` | Document storage bucket |
| `BEDROCK_MODEL_ID` | `anthropic.claude-3-sonnet-20240229-v1:0` | Generation model |
| `BEDROCK_EMBED_MODEL_ID` | `amazon.titan-embed-text-v1` | Embedding model |
| `INDEX_PERSIST_DIR` | `./index_store` | Vector index storage path |
| `LORA_MODEL_PATH` | `./models/finrag-ner-lora` | NER adapter path |
| `RETRIEVER_TOP_K` | `8` | Documents retrieved per query |
| `RERANKER_TOP_N` | `4` | Documents after reranking |

---

## License

MIT
