# 🧠 CLAUDE.md — Multimodal Financial RAG

> Read this first. Everything you need to work on this project is here.

---

## 🎯 What This Project Does

A production RAG system that lets you **ask questions to financial PDFs** and get cited answers in seconds.

- Uploads any financial PDF (annual report, 10-K, earnings release)
- Reads **text AND charts/images** — not just text like most RAG systems
- Returns the exact answer with **page number citations**
- Deployed live on **AWS EKS** and **AWS ECS**

---

## 🌐 Live URLs

| Environment | URL | Status |
|---|---|---|
| **EKS (primary)** | http://finrag.44.206.217.242.nip.io | 🟢 Live |
| **ECS** | http://13.222.137.204:8000 | 🟢 Live |
| **Health check** | GET `/health` | `{"status":"ok","index_loaded":true}` |
| **GitHub** | https://github.com/pritmon/multimodal-finrag | ✅ CI green |

---

## 🏗️ Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **LLM** | Amazon Nova Lite (`amazon.nova-lite-v1:0`) | Works immediately on free tier — no approval needed |
| **Embeddings** | `all-MiniLM-L6-v2` (local) | Free, no throttling, 3 seconds for 800 chunks |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Precise chunk selection after retrieval |
| **Chart Detection** | OpenCLIP `ViT-B-32` | Zero-shot — no training on financial charts needed |
| **PDF Parsing** | PyMuPDF (`fitz`) | Text + images + bounding boxes |
| **API** | FastAPI + Uvicorn | Async, auto-validation, Swagger docs |
| **Container** | Docker multi-stage | Builder 3GB → Runtime 800MB |
| **Orchestration** | AWS EKS + ECS | EKS = K8s production, ECS = simple deploy |
| **Registry** | AWS ECR | `020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api` |
| **Storage** | AWS S3 | Bucket: `pritam-finrag-docs` |
| **CI/CD** | GitHub Actions | ci.yml = auto tests, deploy.yml = manual only |

---

## ☁️ AWS Configuration

| Setting | Value |
|---|---|
| **Account ID** | `020262236277` |
| **Region** | `us-east-1` |
| **EKS Cluster** | `finrag-cluster-eks` |
| **ECS Cluster** | `finrag-cluster` / service: `finrag-service` |
| **IAM Role** | `finrag-task-execution-role` |
| **CloudWatch** | `/ecs/finrag-api` |
| **S3 Bucket** | `pritam-finrag-docs` |

---

## 📁 Project Structure

```
multimodal-finrag/
├── src/
│   ├── api/              # FastAPI app (main.py, schemas.py, static/index.html)
│   ├── ingestion/        # PDF parser, chart extractor, S3 loader
│   ├── rag/              # Pipeline, retriever, embeddings, Bedrock LLM
│   ├── finetune/         # LoRA fine-tuning (dataset, trainer, inference)
│   └── lambda_handler/   # AWS Lambda S3 event handler
├── k8s/                  # Kubernetes manifests (deployment, service, hpa, ingress)
├── scripts/              # build_index.py, train_lora.py, deploy_lambda.py
├── tests/                # pytest suite (test_pdf_parser.py)
├── artifacts/            # Q&A.md, interview.md, AWS_K8S_QNA.md
├── .claude/
│   ├── settings.json     # Permissions + hooks
│   └── commands/         # /test /health /deploy /query /ci-status
├── CLAUDE.md             # ← you are here
└── README.md
```

---

## ⚡ Common Commands

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
kubectl rollout status deployment/finrag-api -n finrag
```

### Health check
```bash
curl http://finrag.44.206.217.242.nip.io/health
curl http://13.222.137.204:8000/health
```

### Query the live API
```bash
curl -X POST http://finrag.44.206.217.242.nip.io/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the revenue?", "top_k": 5}'
```

---

## 🎮 Slash Commands (use in Claude)

| Command | What it does |
|---|---|
| `/test` | Runs the full pytest suite |
| `/health` | Checks both EKS + ECS live deployments |
| `/deploy` | Docker build → ECR push → EKS rollout |
| `/query` | Sends a test question to the live API |
| `/ci-status` | Shows last 5 GitHub Actions runs |

---

## 🔑 Key Design Decisions

> These were deliberate choices. Don't change them without understanding why.

| Decision | Why |
|---|---|
| **Local embeddings** (not Bedrock Titan) | Titan throttled at 5 req/sec → 4+ hours. Local MiniLM = 3 seconds, free |
| **Page-level chunks** (not block-level) | 11,327 blocks → 800 chunks. Indexing: 4 hours → 41 seconds |
| **Nova Lite** (not Claude) | Claude needs payment verification. Nova works immediately |
| **Hybrid search** BM25 + Vector + RRF | BM25 = exact terms. Vector = meaning. Together = best results |
| **asyncio_mode = strict** | "auto" crashes pytest-asyncio 0.23.0 during collection |
| **deploy.yml = workflow_dispatch** | Prevents CI failures when AWS secrets aren't available |
| **Lazy imports in ingestion/__init__.py** | boto3 not available in CI — try/except prevents import failure |
| **is_chat_model = False** on BedrockLLM | LlamaIndex otherwise routes through chat() in Claude format → Nova rejects |

---

## ⚠️ Known Gotchas

- `asyncio_mode` in `pyproject.toml` **must be** `strict` — never change to `auto`
- `deploy.yml` **must** stay as `workflow_dispatch` — if changed to push trigger, CI fails
- EKS nodegroup AMI: use `AL2023_x86_64_STANDARD` — `AL2_x86_64` not supported for K8s 1.34
- EKS subnets: always use the EKS cluster's own VPC subnets, not the default VPC
- Nova API format is **different from Claude** — `{"text": "..."}` not `{"type": "text", "text": "..."}`

---

## 🧪 CI/CD

| Workflow | Trigger | What it runs |
|---|---|---|
| `ci.yml` | Every push to `main` | `pytest tests/test_pdf_parser.py` |
| `deploy.yml` | Manual only (`workflow_dispatch`) | Docker build → ECR → ECS deploy |
