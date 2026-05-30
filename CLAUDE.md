# CLAUDE.md — Multimodal Financial RAG

## What This Project Is
A production RAG (Retrieval-Augmented Generation) system for querying financial PDFs.
- Extracts **text + charts/images** from PDFs (multimodal)
- Indexes content using **sentence-transformers** embeddings + **BM25** hybrid search
- Answers questions using **Amazon Nova Lite** via **AWS Bedrock**
- Deployed live on **AWS EKS** and **AWS ECS**

## Live URLs
- **EKS (primary):** http://finrag.44.206.217.242.nip.io
- **ECS:** http://13.222.137.204:8000
- **Health check:** GET /health → `{"status":"ok","version":"0.1.0","index_loaded":true}`

## Tech Stack
| Layer | Technology |
|-------|-----------|
| LLM | Amazon Nova Lite (`amazon.nova-lite-v1:0`) via AWS Bedrock |
| Embeddings | `all-MiniLM-L6-v2` (local, HuggingFace) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| PDF parsing | PyMuPDF (fitz) |
| API | FastAPI + Uvicorn |
| Container | Docker (multi-stage, non-root user `finrag`) |
| Orchestration | AWS EKS (Kubernetes) + AWS ECS (Fargate) |
| Registry | AWS ECR (`020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api`) |
| Storage | AWS S3 (`pritam-finrag-docs`) |
| CI/CD | GitHub Actions (ci.yml = tests, deploy.yml = manual only) |

## Project Structure
```
src/
  api/          # FastAPI app (main.py, schemas.py, static/index.html)
  ingestion/    # PDF parser, chart extractor, S3 loader
  rag/          # Pipeline, retriever, embeddings, bedrock LLM
  finetune/     # LoRA fine-tuning (dataset, trainer, inference)
  lambda_handler/ # AWS Lambda handler
scripts/        # build_index.py, train_lora.py, deploy_lambda.py
k8s/            # Kubernetes manifests (deployment, service, hpa, ingress)
tests/          # pytest tests (test_pdf_parser.py)
artifacts/      # interview.md, QNA.md
```

## Common Commands

### Run locally
```bash
source .venv/bin/activate
uvicorn src.api.main:app --reload --port 8000
```

### Run tests
```bash
source .venv/bin/activate
pytest tests/test_pdf_parser.py -v
```

### Build & push Docker image
```bash
docker build -t finrag-api .
docker tag finrag-api:latest 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 020262236277.dkr.ecr.us-east-1.amazonaws.com
docker push 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest
```

### Deploy to EKS
```bash
kubectl apply -f k8s/
kubectl rollout restart deployment/finrag-api -n finrag
```

### Health check
```bash
curl http://finrag.44.206.217.242.nip.io/health
curl http://13.222.137.204:8000/health
```

### Query the API
```bash
curl -X POST http://finrag.44.206.217.242.nip.io/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the revenue?", "top_k": 5}'
```

## AWS Config
- **Account:** 020262236277
- **Region:** us-east-1
- **EKS cluster:** finrag-cluster-eks
- **ECS cluster:** finrag-cluster / service: finrag-service
- **IAM role:** finrag-task-execution-role
- **CloudWatch log group:** /ecs/finrag-api

## Key Design Decisions
1. **Hybrid search:** Dense (embeddings) + sparse (BM25) → reranked by cross-encoder
2. **Multimodal:** Charts/images extracted and captioned via Bedrock Vision
3. **No llama-index-retrievers-bm25:** Uses `rank_bm25` directly (avoids dep conflict)
4. **Docker multi-stage:** Builder stage installs deps, runtime is slim
5. **asyncio_mode=strict** in pyproject.toml (not "auto") to avoid pytest-asyncio crash

## CI/CD
- **ci.yml** — triggers on every push to main, runs `tests/test_pdf_parser.py` only
- **deploy.yml** — manual trigger only (`workflow_dispatch`), requires AWS secrets in GitHub

## GitHub
- **Repo:** https://github.com/pritmon/multimodal-finrag
- **CI badge:** should be green ✅
