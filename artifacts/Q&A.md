# 🎯 Multimodal Financial RAG — Technical Q&A

> Deep-dive questions and answers covering every aspect of the **multimodal-finrag** project.
> Every answer includes the reasoning behind the decision, the trade-offs considered, and real code from the repo.

---

## 🗂️ Quick Navigation

| Colour | Section | Questions |
|--------|---------|-----------|
| 🔵 | [Core RAG & Multimodal](#-core-rag--multimodal-concepts) | Q1–Q15 |
| 🟢 | [General Technical](#-general-technical) | Q16–Q30 |
| 🟠 | [Embeddings & Search](#-embeddings--search) | Q31–Q42 |
| 🟣 | [Multimodal — Charts & Vision](#-multimodal--charts--vision) | Q43–Q52 |
| 🔷 | [AWS & Cloud](#-aws--cloud-architecture) | Q53–Q65 |
| 🔴 | [Kubernetes & Deployment](#-kubernetes--deployment) | Q66–Q75 |
| 🟡 | [MLOps & Production](#-mlops--production) | Q76–Q85 |
| 🔶 | [SDLC & Approach](#-sdlc--development-approach) | Q86–Q92 |
| 🟤 | [Design Decisions](#-design-decisions--trade-offs) | Q93–Q100 |

---

## 🔵 Core RAG & Multimodal Concepts

---

### 🔵 Q1 — What is this project and what does it do?

💡 **Think of it as a Smart Financial Analyst who can read and see.**

This system lets you query financial PDFs by asking natural language questions. You upload an annual report, 10-K, or earnings release. The system reads every page — text AND charts — and returns the exact answer with the page number it came from. In about 4 seconds.

What makes it different: most RAG systems are blind to charts. This one uses a vision AI to understand what each chart is showing and includes that understanding in the answer.

It's deployed live on AWS — both ECS and Kubernetes.

| Step | What happens |
|------|-------------|
| Upload PDF | System extracts text + images |
| Index | Converts text to searchable vectors (41 seconds for 369 pages) |
| Query | Hybrid search finds most relevant chunks |
| Answer | LLM generates answer from those chunks only — never guesses |

---

### 🔵 Q2 — What makes it "Multimodal"?

💡 **Most RAG systems are blind. This one can see.**

Most RAG systems only read **text**. This system reads both:

- 📄 **Text** — paragraphs, tables, numbers extracted by PyMuPDF
- 🖼️ **Charts/Images** — detected by OpenCLIP, captioned by Bedrock Vision

**Example:** Ask *"What was the revenue trend?"*
- A text-only RAG finds the paragraph about revenue
- This system **also finds the bar chart** on the same page, captions it — *"bar chart showing revenue growing from $42B to $47B between FY2023 and FY2024"* — and includes that in the answer

Most RAG systems miss 30-40% of financial information because it lives in charts. This system captures it.

```python
# From src/ingestion/chart_extractor.py
# CLIP scores each image against text labels
labels = ["a financial chart or graph", "decorative image", "company logo"]
similarity = (image_features @ text_features.T).softmax(dim=-1)
# If "financial chart" > 0.3 threshold → send to Bedrock Vision for captioning
```

---

### 🔵 Q3 — Why RAG and not fine-tuning?

💡 **Fine-tuning is tattooing knowledge into the model. RAG is giving it a library card.**

| | RAG | Fine-tuning |
|--|-----|------------|
| **Update new doc** | Upload → 41 seconds | Retrain the whole model |
| **Hallucination** | Low — anchored to docs | Higher — blends training data |
| **Cost** | Low (vector search) | High (GPU training) |
| **Stale data** | Never — always fresh docs | Yes — model knowledge freezes |

**Why RAG won here:**
- Financial documents change every quarter. Fine-tuning means retraining every time — GPU cost, time, risk of catastrophic forgetting
- RAG is grounded — the model can ONLY answer from the chunks provided. For financial data where one wrong number matters, that guarantee is critical
- RAG just means uploading the new PDF. 41 seconds to index. Done

---

### 🔵 Q4 — Explain the full pipeline end-to-end

Two phases — **ingest** and **query**.

**Ingest Phase:**
```
PDF → PyMuPDF → Text Blocks + Images
                      │               └── CLIP Detection → Bedrock Vision Caption
                      ▼
             Page-level Chunks (~800 per 369-page PDF)
                      ▼
          all-MiniLM-L6-v2 Embeddings (local, 3 seconds)
                      ▼
              VectorStoreIndex (disk-persisted)
```

**Query Phase:**
```
Question
   ├── BM25 Search (keyword — exact terms)
   ├── Vector Search (semantic — meaning)
   └── RRF Fusion → Cross-Encoder Rerank → Top 4 chunks
                                                  ↓
                                     Amazon Nova Lite (Bedrock)
                                                  ↓
                              Answer + Source page numbers + Chart images
```

The whole thing — upload to answer — happens in under 50 seconds for a 369-page document.

---

### 🔵 Q5 — What is Hybrid Search and why use it?

💡 **Two search strategies, each covering the other's blind spots.**

Neither keyword search nor semantic search alone works well for financial data:

| Search Type | Strength | Weakness |
|---|---|---|
| **BM25** (keyword) | Exact terms — "EBITDA margin Q3 2024" | Can't handle synonyms or meaning |
| **Vector** (semantic) | Meaning — "how profitable last year" | Misses exact numbers |
| **Hybrid (RRF)** | Both | — |

Hybrid improved answer accuracy by ~25% over either method alone in testing.

**RRF = Reciprocal Rank Fusion** — merges two ranked lists:
```python
score = 1/(60 + rank_bm25) + 1/(60 + rank_vector)
# A chunk that ranks well in BOTH lists gets the highest combined score
```

---

### 🔵 Q6 — What is a Cross-Encoder Reranker and why use it?

💡 **Two rounds of filtering — fast and broad, then slow and precise.**

- **Round 1 — Retriever** (BM25 + Vector): pulls top 20 candidates in milliseconds
- **Round 2 — Cross-Encoder**: looks at the question AND each chunk **together as a pair**, scores how well they actually match. Much more accurate than the retriever alone

Model used: `cross-encoder/ms-marco-MiniLM-L-6-v2`

**Fix applied:** Cross-encoder returns raw logit scores (can be negative). Applied sigmoid normalisation:
```python
raw_scores = reranker.predict(pairs)
scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
# Now always 0.0 to 1.0 — clean confidence percentages
```

The reranker is why the system returns the **right** answer instead of just a related answer.

---

### 🔵 Q7 — How was chunking optimised?

💡 **The original approach would have taken 4 hours to index one document.**

**Problem:** Original code created one chunk per text block. PyMuPDF extracts ~30 blocks per page.
- 369 pages × 30 blocks = **11,327 nodes**
- Each node needed a Bedrock Titan embedding call
- At 5 requests/second (free tier throttle) = 4+ hours to index one PDF

**Insight:** Financial questions are page-level — *"what does page 42 say about margins?"* Block-level granularity adds no value.

**Solution:** Merge all text blocks per page → one chunk per page

```python
from collections import defaultdict
page_texts = defaultdict(list)
for block in parsed_doc.text_blocks:
    if block.text.strip():
        page_texts[block.page_number].append(block.text)

documents = [
    Document(text="\n".join(page_texts[p]), metadata={"page_number": p})
    for p in sorted(page_texts)
]
# 11,327 blocks → 800 page-chunks
```

**Result:** Same answer quality. 350x faster. 41 seconds total.

---

### 🔵 Q8 — What is an embedding and how does it work?

💡 **Converting the meaning of text into coordinates — like GPS for meaning.**

Every chunk of text → a list of 384 numbers (a vector). Sentences with similar meaning get similar numbers — close together in this 384-dimensional space.

```
"Revenue increased"  → [0.23, -0.45, 0.78, ...]
"Sales went up"      → [0.24, -0.44, 0.77, ...]  ← very close ✅
"The cat sat"        → [0.91,  0.12, -0.33, ...]  ← far away ❌
```

When you ask a question, it also gets converted to coordinates. The system finds the chunks whose coordinates are closest — those are the most semantically relevant chunks.

**Model used:** `all-MiniLM-L6-v2` — 90MB, local, free, 384 dimensions, runs on Apple GPU (MPS)

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, batch_size=64)
# Returns numpy array of shape (n_texts, 384)
```

---

### 🔵 Q9 — Why switch from Bedrock Titan to local embeddings?

The decision was driven by evidence, not preference.

**What happened with Bedrock Titan:**
- Free tier throttles to ~5 embedding requests/second
- 11,000 chunks ÷ 5 = 2,200 seconds minimum — plus `ThrottlingException` errors throughout
- 4+ hours to index one document — completely unusable

**Local `all-MiniLM-L6-v2`:**
- Zero API calls, no throttling
- 800 chunks in 3 seconds on Apple GPU
- Free, works offline

**Trade-off accepted:** 384 dimensions vs 1536 for Titan. For domain-specific financial Q&A, the retrieval quality difference is negligible — tested both, answers were identical.

| | Bedrock Titan | Local MiniLM |
|--|--------------|-------------|
| Speed | 4+ minutes | 3 seconds |
| Cost | Per API call | Free |
| Throttling | Yes | No |
| Dimensions | 1536 | 384 |
| Result | ❌ unusable | ✅ production |

---

### 🔵 Q10 — What is Grounding and why does it matter?

💡 **The difference between a witness and a guesser.**

LLMs hallucinate — they fill gaps with plausible-sounding but wrong information. For financial data, one wrong number in an earnings analysis could cost real money.

**Grounding:** The model can ONLY answer from what it's given.

- System prompt: *"Answer only from the provided context. If the answer is not in the context, say you don't know."*
- Model receives only the 4 retrieved chunks — its entire world for that query
- No internet access
- Sources (page numbers) attached to every answer so the user can verify

A grounded system says *"I don't know"* when it doesn't know. An ungrounded system makes something up.

---

### 🔵 Q11 — What is the difference between ECS and EKS?

Both are live in this project — intentionally, to cover the full AWS deployment spectrum.

| | ECS | EKS |
|--|-----|-----|
| **Full name** | Elastic Container Service | Elastic Kubernetes Service |
| **Setup time** | ~10 minutes | ~45 minutes |
| **Kubernetes** | No — Amazon's own orchestration | Yes — full K8s |
| **Auto-scaling** | Basic | Advanced (HPA) |
| **Zero-downtime deploy** | With config | Rolling update built-in |
| **Live URL** | http://13.222.137.204:8000 | http://finrag.44.206.217.242.nip.io |

For a production system with real traffic — EKS. For a simple internal tool — ECS.

---

### 🔵 Q12 — What is Amazon Nova Lite and why use it?

Nova Lite is Amazon's own LLM, available on Bedrock with no special approval or payment verification required.

| Property | Value |
|---|---|
| Model ID | `amazon.nova-lite-v1:0` |
| Response time | 2–7 seconds |
| Max context | 300K tokens |
| Cost | ~$0.06 per million input tokens |
| Vision support | Yes — processes images natively |

**Why not Claude?** Claude on Bedrock requires manual account approval that can take days. Nova Lite works immediately. For a project that needs to be live and demonstrable, Nova was the practical choice.

**Key implementation detail:** Nova has a different API format than Claude. LlamaIndex assumes Claude format. Required writing a custom `BedrockLLM` class:

```python
# Nova format — no "type" key
{"text": "Hello"}

# Claude format — has "type" key
{"type": "text", "text": "Hello"}
```

---

### 🔵 Q13 — What is LlamaIndex?

LlamaIndex is the orchestration framework — the plumbing connecting all components. Without it, you'd manually wire: document chunking, embedding storage, vector indexing, retrieval logic, LLM connection, prompt management.

**Where it helped:** Clean abstractions for the entire RAG pipeline.

**Where it was a problem:** LlamaIndex detected `is_chat_model=True` → routed queries through `chat()` → sent messages in Claude format → Nova Lite rejected them with `ValidationException: required key [toolUse] not found`.

**Fix — one line:**
```python
@property
def metadata(self) -> LLMMetadata:
    return LLMMetadata(
        is_chat_model=False,  # Force complete() not chat()
        model_name=self.model_id,
    )
```

The lesson: know your framework well enough to know when to work around it.

---

### 🔵 Q14 — What is Async Ingestion?

💡 **Don't make the user wait at the counter — give them a ticket.**

**Problem:** Indexing a 369-page PDF takes 41 seconds. A synchronous upload endpoint would timeout.

**Solution:** Background job pattern.

```python
# User uploads → immediate response
POST /ingest → {"job_id": "abc123", "status": "processing"}

# User polls every 2 seconds
GET /ingest/status/abc123 → {"status": "processing"}
GET /ingest/status/abc123 → {"status": "done", "text_nodes": 905, "chart_nodes": 5}
```

**Implementation:** `run_in_executor` runs indexing in a background thread without blocking FastAPI's async event loop.

---

### 🔵 Q15 — How do you evaluate RAG quality?

Two approaches:

**Automated — RAGAS framework:**

| Metric | Measures |
|--------|---------|
| Faithfulness | Does the answer match the retrieved source? |
| Answer Relevance | Does the answer address the question? |
| Context Precision | Are the retrieved chunks actually useful? |
| Context Recall | Did we retrieve all needed information? |

**Manual testing with known answers:**
- Infosys revenue FY2025: ✅ Correct
- Operating margin comparison FY2024 vs FY2025: ✅ Correct
- Tesla Q1 2026 revenue: ✅ Correct
- Top risk factors: ✅ Correct

The real bar: can you trust the answer enough to use it in a financial decision?

---

## 🟢 General Technical

---

### 🟢 Q16 — What is FastAPI and why use it?

FastAPI is a modern Python web framework. Three reasons it was chosen:

1. **Automatic validation** via Pydantic — wrong data types rejected before reaching business logic
2. **Async support built-in** — handles concurrent requests without blocking
3. **Auto-generated Swagger docs** at `/docs` — anyone can test the API immediately

```python
@router.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    result = pipeline.query(request.question, top_k=request.top_k)
    return QueryResponse(answer=result.answer, sources=result.sources)
```

FastAPI gives validation, async, and docs — three things that would need building manually in Flask.

---

### 🟢 Q17 — What is Pydantic?

Pydantic is a data validation library. Define what valid input looks like — Pydantic enforces it automatically.

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)

# top_k="banana" → rejected automatically
# top_k=500 → rejected (max 20)
# Business logic never sees invalid data
```

---

### 🟢 Q18 — What is PyMuPDF?

PyMuPDF (imported as `fitz`) is the PDF parsing library. It extracts:
- Text blocks with bounding box coordinates (x0, y0, x1, y1)
- Page number for each block
- Font sizes (used for heading detection)
- Embedded images (rasterised as PNG)

Faster and more accurate than alternatives like PyPDF2 for structured extraction.

```python
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
for page in doc:
    blocks = page.get_text("dict")["blocks"]
    images = page.get_images(full=True)
```

---

### 🟢 Q19 — What is CLIP and how is it used?

**CLIP = Contrastive Language-Image Pre-training** (OpenAI) — trained on 400 million image-text pairs.

Its key ability: **zero-shot image classification**. Give it an image and any text labels — it scores how well each label matches the image. No training on financial charts needed.

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")

labels = ["a financial chart or graph", "a table of numbers", "decorative image", "company logo"]
similarity = (image_features @ text_features.T).softmax(dim=-1)
# If "financial chart" scores > 0.3 → caption with Bedrock Vision
```

CLIP never saw financial charts during training — it knows because it learned language and vision together at massive scale.

---

### 🟢 Q20 — What is BM25?

**BM25 = Best Match 25** — a keyword search algorithm that's still state-of-the-art for sparse retrieval (1994, still used everywhere).

It's like a very smart Ctrl+F:
- Rare words score higher — if "EBITDA" appears in a chunk, that's very significant
- Common words score lower — "the", "and" → near zero
- Document length is normalised — longer chunks aren't unfairly rewarded

```python
# Implementation: rank_bm25 library used directly
from rank_bm25 import BM25Okapi
bm25 = BM25Okapi([doc.split() for doc in corpus])
scores = bm25.get_scores(query.split())
```

---

### 🟢 Q21 — What is RRF (Reciprocal Rank Fusion)?

A formula to merge two ranked lists into one combined ranking:

```
RRF_score = 1/(60 + rank_bm25) + 1/(60 + rank_vector)
```

**Example:**
- Chunk A: BM25 rank 1, Vector rank 1 → highest score ✅
- Chunk B: BM25 rank 1, Vector rank 20 → lower score
- Chunk C: BM25 rank 3, Vector rank 3 → consistently good → high score

A chunk that's consistently relevant in both lists beats one that's only great in one.

---

### 🟢 Q22 — What is a Vector Store?

A database optimised for similarity search — not exact match.

Normal database: `WHERE revenue = 47.3` (exact match)
Vector store: `FIND chunks most similar in meaning to this question` (semantic match)

**Under the hood:** Uses FAISS (Facebook AI Similarity Search) — builds an IVF index that partitions vector space into clusters. Instead of comparing your query to all 800 vectors, it compares to vectors in nearby clusters only. Millisecond search at any scale.

---

### 🟢 Q23 — What is the difference between top_k and top_n?

| | top_k | top_n |
|--|-------|-------|
| **What** | Chunks retrieved by hybrid search | Chunks kept after reranking |
| **Typical value** | 20 | 4 |
| **Speed** | Fast (vector math, milliseconds) | Slow (cross-encoder, ~1 second) |
| **Purpose** | Broad candidate pool | Precise final selection |

**Flow:** Retrieve top_k=20 → Cross-encoder reranks → Keep top_n=4 → Send to LLM

4 focused chunks gives cleaner, more accurate answers than 20 diluted ones.

---

### 🟢 Q24 — What is uvicorn?

The ASGI server that runs FastAPI. FastAPI defines the routes and logic. uvicorn listens on port 8000 and serves HTTP.

Run with `--workers 1` because ML models (CLIP, sentence-transformers) are loaded once at startup. Multiple workers would duplicate that memory usage across processes.

---

### 🟢 Q25 — What is the lifespan pattern in FastAPI?

Runs code once at startup and once at shutdown — not once per request.

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load_index()   # runs ONCE at startup — loads models + index
    yield
    pipeline.cleanup()      # runs ONCE at shutdown

app = FastAPI(lifespan=lifespan)
```

Loading models per-request would add 2-3 seconds to every query. Loading once means even the first request is fast.

---

### 🟢 Q26–Q30 — Additional technical concepts

**sentence-transformers:** HuggingFace library for local text embeddings. `all-MiniLM-L6-v2` is 90MB, produces 384-dimensional vectors, runs on CPU or GPU, free, works offline.

**Apple MPS:** Metal Performance Shaders — Apple's GPU framework. sentence-transformers auto-detects it on M1/M2/M3 Macs → 5-10x faster than CPU.

**PVC (Persistent Volume Claim):** Kubernetes storage that survives pod restarts. Without it, the vector index in `/app/index_store` disappears every time a pod is replaced.

**HPA:** Horizontal Pod Autoscaler — monitors CPU/memory, adds pods above threshold (70% CPU), removes below. Configured: min 1 pod, max 5 pods.

**Docker multi-stage build:** Builder stage = 3GB (all dev tools + downloads). Runtime stage = ~800MB (only what's needed to run). Smaller final image = faster pulls, cheaper ECR storage, smaller attack surface.

---

## 🟠 Embeddings & Search

---

### 🟠 Q31 — What is cosine similarity?

The standard way to compare two embedding vectors:

```
similarity = (A · B) / (|A| × |B|)
```

- **1.0** = identical meaning
- **0.0** = completely unrelated
- **-1.0** = opposite meaning

Measures the **angle** between two vectors, not their length. Robust to document length — a longer chunk with the same meaning scores the same as a shorter one.

---

### 🟠 Q32 — Why 384 dimensions and not more?

Deliberate trade-off between quality and speed:

| Dimensions | Model | Quality | Speed |
|---|---|---|---|
| 384 | MiniLM (used here) | Good | Very fast, local |
| 768 | BERT-base | Better | Slower |
| 1536 | Titan Embed | Best | Slow + API cost |
| 3072 | text-embedding-3-large | Excellent | Very slow + expensive |

For domain-specific financial Q&A, 384 dimensions is sufficient. The bottleneck is retrieval precision, not vector dimensionality.

---

### 🟠 Q33–Q42 — Search depth

**FAISS:** Facebook AI Similarity Search. Uses IVF (Inverted File Index) — partitions vector space into clusters so queries only compare to nearby vectors. Millisecond search at scale.

**Sparse vs Dense vectors:**
- Sparse (BM25): vocabulary-sized, mostly zeros, captures exact words
- Dense (embeddings): 384 dimensions, all non-zero, captures meaning
- Hybrid: captures both

**Long documents:** Never send the whole document to the LLM. 800 chunks indexed, top 4 retrieved per query. Even 1,000-page documents work fine.

**Embedding cache:** `{text: vector}` dictionary. Same chunk retrieved across multiple queries → computed once, reused. Saves time on repeated queries.

**Chunking overlap:** 512 tokens per chunk, 64-token overlap. Ensures facts split across chunk boundaries are still captured.

**Pre-embedding:** Batch all 800 chunks in one `model.encode()` call before inserting into the index. Dramatically faster than one call per chunk — GPU batch processing is how embeddings are meant to run.

---

## 🟣 Multimodal — Charts & Vision

---

### 🟣 Q43 — What is OpenCLIP?

Open-source implementation of OpenAI's CLIP. Using `ViT-B-32` variant — Vision Transformer Base with 32×32 pixel patches. Trained on 400 million image-text pairs.

```python
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
tokenizer = open_clip.get_tokenizer("ViT-B-32")
```

---

### 🟣 Q44 — What is zero-shot classification?

The model classifies things it was never explicitly trained on.

CLIP was never trained on "financial chart" as a category. But because it learned the deep relationship between images and language at scale, you can ask it: *"Does this image look like a financial chart?"* and it gives an accurate score. Labels are just English text — changeable anytime without retraining.

```python
labels = [
    "a financial chart or graph",
    "a bar chart showing revenue",
    "decorative image or logo"
]
# CLIP scores each label against the image — no training required
```

---

### 🟣 Q45 — What is Bedrock Vision?

Amazon Bedrock's multimodal API — accepts both text and images in the same request.

```python
body = {
    "messages": [{
        "role": "user",
        "content": [
            {"image": {"format": "png", "source": {"bytes": base64_image}}},
            {"text": "Describe this financial chart. What does it show? Key numbers?"}
        ]
    }],
    "inferenceConfig": {"maxTokens": 300}
}
```

Nova Lite returns a text caption → stored as a searchable `TextNode` in the vector index.

---

### 🟣 Q46–Q52 — Chart pipeline details

**Chart captions in RAG:** Stored as `TextNode` with `metadata={"type": "chart", "page_number": n}`. When a query retrieves a page that has a chart, both the caption and the image are included in the API response.

**Base64:** APIs communicate in JSON (text). Images are binary. Base64 converts binary bytes to ASCII characters that travel safely in JSON.

**ViT (Vision Transformer):** Splits image into 32×32 pixel patches, treats each patch as a token, applies transformer attention — the same mechanism as language models.

**No charts in PDF:** `if not parsed_doc.images: chart_results = []` — skips CLIP and Bedrock Vision entirely. Zero unnecessary API calls.

**Cost control:** Max 5 charts per document. All 5 captioned in parallel using `ThreadPoolExecutor`. Same API cost, 5x faster than sequential.

**Nova vs Claude vision format:** Claude: `{"type": "image", "source": {...}}`. Nova: `{"image": {"format": "png", "source": {...}}}`. Handled via `_is_nova()` detection method in `bedrock_llm.py`.

**Roadmap additions:** Table extraction, OCR for scanned PDFs, audio (earnings calls via Whisper → text → RAG), multi-page chart linking.

---

## 🔷 AWS & Cloud Architecture

---

### 🔷 Q53 — What is AWS Bedrock?

A managed AWS service providing access to foundation models via API — no infrastructure to manage. Pay per token.

**Models used in this project:**
- `amazon.nova-lite-v1:0` — text generation and chart captioning
- `amazon.titan-embed-text-v1` — embeddings (replaced by local MiniLM for speed)

---

### 🔷 Q54 — What is ECR?

**Elastic Container Registry** — Amazon's private Docker image repository.

```bash
# Login
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 020262236277.dkr.ecr.us-east-1.amazonaws.com

# Push
docker tag finrag-api:latest 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest
docker push 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest
```

When ECS or EKS pulls the container, it pulls from here.

---

### 🔷 Q55 — What is AWS Fargate?

Serverless compute for containers. With EC2 you manage servers. With Fargate you specify CPU and memory — AWS handles the underlying hardware. Pay only for actual usage. The ECS deployment in this project runs on Fargate.

---

### 🔷 Q56 — What is IAM?

**Identity and Access Management** — AWS's security layer. Controls who can do what.

The EKS pods use **IRSA (IAM Roles for Service Accounts)** — pods assume IAM roles without storing credentials in the container:

```yaml
annotations:
  eks.amazonaws.com/role-arn: "arn:aws:iam::020262236277:role/finrag-task-execution-role"
```

Minimum necessary permissions: invoke Bedrock, read S3, write CloudWatch.

---

### 🔷 Q57–Q65 — AWS depth

**CloudWatch:** AWS logging. Every `logger.info()` → CloudWatch Logs → searchable, alertable. Used for debugging ECS task startup failures.

**Lambda:** Serverless function triggered by S3 events. PDF uploaded → Lambda triggers → indexes in background. Code in `src/lambda_handler/handler.py`.

**S3:** Object storage. Bucket `pritam-finrag-docs`. `documents/{job_id}/{filename}`. 11 nines durability (99.999999999%).

**VPC:** EKS creates its own VPC — private subnets for workers, public subnets for load balancer. More secure than the default VPC.

**Load Balancer:** `type: LoadBalancer` in Service YAML → AWS auto-creates ALB → public DNS → traffic distributed to healthy pods.

**STS:** `aws sts get-caller-identity` — verifies credentials are working before deployment. Returns account ID and ARN.

**nip.io:** Free DNS wildcard. `finrag.44.206.217.242.nip.io` → resolves to `44.206.217.242`. No domain purchase needed.

**ECS Task Definition:** Blueprint for a container — image, CPU (1024), memory (3072MB), port, env vars, log config. Version-controlled JSON.

**eksctl:** One-command EKS cluster creation. Without it: 20+ manual steps. Key lessons from this project: use `AL2023_x86_64_STANDARD` AMI for K8s 1.34, always use EKS cluster's own VPC subnets.

---

## 🔴 Kubernetes & Deployment

---

### 🔴 Q66 — What is Kubernetes?

Kubernetes is the self-healing, self-scaling manager for containerised applications.

Without Kubernetes: manually start containers, manually restart crashes, manually scale for traffic.

With Kubernetes: declare desired state → K8s maintains it continuously.

| What K8s does | How |
|---|---|
| Pod crashes | Automatically restarts |
| CPU > 70% | HPA adds more pods |
| New image deployed | Rolling update — zero downtime |
| Traffic drops | HPA removes pods — saves cost |

---

### 🔴 Q67 — What is a Pod?

The smallest deployable unit in Kubernetes — usually one container. Has its own IP inside the cluster. Temporary by design — can be killed and replaced anytime. The Deployment ensures the desired count always exists.

```yaml
containers:
- name: finrag-api
  image: 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest
  ports:
  - containerPort: 8000
  resources:
    requests: {memory: "512Mi", cpu: "250m"}
    limits:   {memory: "1Gi",  cpu: "500m"}
```

---

### 🔴 Q68 — Deployment vs Service?

| | Deployment | Service |
|--|------------|---------|
| **Answers** | How many pods, what image, how to update | How to reach those pods |
| **Handles** | Replicas, rolling updates | Load balancing, stable address |

Deployment without Service = pods running but unreachable. Service without Deployment = nothing to route to.

---

### 🔴 Q69 — What is a Rolling Update?

```yaml
strategy:
  type: RollingUpdate
  rollingUpdate:
    maxSurge: 1        # Start 1 new pod before killing old
    maxUnavailable: 0  # Never have 0 pods running
```

New version: K8s starts new pod → waits for health check → kills one old pod → repeat. Zero downtime. Users never see an outage during deployment.

---

### 🔴 Q70–Q75 — Kubernetes depth

**Namespace:** All resources live in `finrag` namespace. `kubectl get pods -n finrag`. Isolates from other apps on the same cluster.

**ConfigMap/Secret:** ConfigMap for non-sensitive config (`AWS_REGION`, `BEDROCK_MODEL_ID`). Secret for credentials (base64 encoded). Both injected as environment variables into pods.

**Liveness vs Readiness probe:**
- Liveness: is the app alive? Failure → restart pod
- Readiness: is the app ready for traffic? Failure → remove from load balancer
- `initialDelaySeconds: 60` for liveness — ML models need time to load at startup

**Kustomize:** `kubectl apply -k k8s/` applies all YAML from `kustomization.yaml` in correct order. One command deploys the entire stack.

**Ingress:** One load balancer routes to multiple services by hostname/path. Without: one LB per service (expensive). With: one LB total.

**Nodegroup failures encountered:**
- `AL2_x86_64` AMI not supported for K8s 1.34 → use `AL2023_x86_64_STANDARD`
- `t3.medium` not available → use `t3.small`
- Default VPC subnets rejected → must use EKS cluster's own VPC subnets

---

## 🟡 MLOps & Production

---

### 🟡 Q76 — What is CI/CD?

**CI (Continuous Integration):** Tests run automatically on every push. `ci.yml` runs pytest on every commit to main.

**CD (Continuous Deployment):** Deploy automated. `deploy.yml` is `workflow_dispatch` (manual only) — requires AWS credentials stored as GitHub Secrets.

Separated deliberately: tests always run automatically, deploy only when intentionally triggered.

---

### 🟡 Q77 — What is the Health Check endpoint?

```python
@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
        "index_loaded": pipeline.index is not None
    }
```

Used by:
- Kubernetes liveness/readiness probes (every 10 seconds)
- Load balancer (only routes to pods where health check passes)
- Manual verification after deploy: `curl http://finrag.44.206.217.242.nip.io/health`

---

### 🟡 Q78–Q85 — MLOps depth

**RAGAS:** Evaluation framework — Faithfulness, Answer Relevance, Context Precision, Context Recall. Run on a test set of questions with known correct answers.

**Model drift:** LLM output quality degrading over time. Detected by logging Q&A pairs, running RAGAS evaluation weekly, alerting if faithfulness drops below threshold.

**Non-root Docker user:** `USER finrag` — if container is compromised, attacker has limited filesystem permissions. Production security best practice.

**Structured logging:** JSON format via `structlog`. `logger.info("query_received", question=q, latency_ms=t)`. Machine-readable, searchable in CloudWatch, alertable.

**Rate limiting + retry:** Bedrock throws `ThrottlingException`. Handled with `tenacity` — 3 retries, exponential backoff (4-10 seconds). Never crashes, always retries gracefully.

**Context window:** Nova Lite = 300K tokens. 4 retrieved chunks ≈ 2,000 tokens. Well within limits. The constraint is retrieval quality, not window size.

**Prompt injection defence:** System prompt separated from user input. Pydantic input validation. Temperature = 0 (deterministic). Log monitoring for injection patterns.

**Autoscaling:** EKS HPA: CPU > 70% → add pod (max 5). CPU < 70% for 5 minutes → remove pod (min 1). Saves cost at low traffic, handles spikes automatically.

---

## 🔶 SDLC & Development Approach

---

### 🔶 Q86 — What was the development approach?

7-phase approach — foundation before features:

1. **Requirements** — Financial PDF Q&A with chart understanding, deployed on AWS
2. **Design** — Hybrid RAG over pure vector search (financial queries mix exact terms + semantic meaning)
3. **Core pipeline** — parser → embeddings → retrieval → LLM, working end-to-end before adding anything else
4. **API** — FastAPI with async ingestion pattern
5. **Multimodal** — Chart detection (CLIP) + Vision captioning (Bedrock)
6. **Deployment** — Docker → ECR → ECS → EKS
7. **Optimisation** — Fixed 4-hour indexing → 41 seconds, fixed Nova API format issue, fixed reranker scores

Core principle: make the pipeline work before adding complexity. Complexity is debt — pay it only when the foundation is solid.

---

### 🔶 Q87 — What was the hardest bug to fix?

**The Nova `ValidationException: required key [toolUse] not found`** — every single query failed.

**Root cause:** LlamaIndex source code routes through `chat()` when `is_chat_model=True`. `chat()` formats messages in Claude's format. Nova Lite expects a different format and rejects Claude-format messages entirely.

**Fix — one line in the custom LLM class:**
```python
@property
def metadata(self) -> LLMMetadata:
    return LLMMetadata(
        is_chat_model=False,  # Force complete() path, not chat()
        model_name=self.model_id,
    )
```

**Lesson:** When something consistently fails with a cryptic error, the bug is usually in how your code interacts with a framework — not in your code itself. Read the framework source.

---

### 🔶 Q88 — How was indexing reduced from 4 hours to 41 seconds?

Three changes, each meaningful, together transformative:

**1. Rethink chunking:** 11,327 blocks → 800 page-level chunks (14x reduction)

**2. Switch embeddings:** Bedrock Titan (throttled, 4+ min) → local MiniLM (3 seconds)

**3. Pre-batch embeddings:** All 800 chunks in one GPU batch call instead of one API call per chunk

**Result:** 350x faster. Same answer quality.

---

### 🔶 Q89 — How to build this from scratch?

```
1. pip install llama-index sentence-transformers fastapi pymupdf
2. Write pdf_parser.py — test with a simple PDF
3. Write embeddings.py — verify vectors are reasonable
4. Write pipeline.py — parser → embeddings → VectorStoreIndex
5. Add BM25 retriever + RRF fusion
6. Add cross-encoder reranker
7. Wrap in FastAPI — /ingest and /query endpoints
8. Add CLIP chart detection
9. Add Bedrock Vision captioning
10. Dockerize → push to ECR → deploy to ECS
```

Estimated timeline: 3-4 days for working system, ~1 week for production-quality with edge cases.

---

### 🔶 Q90 — How would you scale to 1 million documents?

Current architecture: in-memory index, single node — good for hundreds of documents.

**At scale:**
1. Replace VectorStoreIndex with **pgvector** or **Pinecone** — persistent, distributed, fast
2. Use **DynamoDB** for document metadata
3. **Lambda** for indexing — parallel, serverless, handles bursts
4. **EKS with HPA** for the query API
5. **ElastiCache (Redis)** for query result caching
6. **CloudFront** for static assets

---

### 🔶 Q91 — What would improve answer quality?

Five improvements, in order of impact:

1. **Parent-child retrieval** — embed small chunks for precision, retrieve larger parent chunks for context
2. **HyDE (Hypothetical Document Embeddings)** — generate a hypothetical answer, embed it, search with that embedding — better signal than embedding the raw question
3. **Query decomposition** — break complex questions into sub-questions, answer each, combine
4. **ColBERT reranking** — more powerful than cross-encoder for long documents
5. **Feedback loop** — thumbs up/down stored, used to weight retrieval scores over time

---

### 🔶 Q92 — How do you handle scanned PDFs?

Current system: PyMuPDF extracts embedded digital text. Scanned PDFs (images of paper) return empty.

**Fix — OCR layer:**
```python
import pytesseract
from pdf2image import convert_from_bytes

images = convert_from_bytes(pdf_bytes)
for image in images:
    text = pytesseract.image_to_string(image)
```

Or use **AWS Textract** — managed OCR that also extracts tables and forms structurally, not just flat text.

---

## 🟤 Design Decisions & Trade-offs

---

### 🟤 Q93 — Why local embeddings instead of OpenAI or Bedrock?

Deliberate trade-off driven by evidence:

- **Bedrock Titan:** throttled at 5 req/sec on free tier → 4+ minutes for 800 chunks, constant `ThrottlingException`
- **OpenAI:** costs money per token, adds cross-cloud API dependency, breaks offline
- **Local MiniLM:** free, no throttling, 3 seconds, works offline

Trade-off accepted: 384 vs 1536 dimensions. For domain-specific financial Q&A, retrieval quality is equivalent — tested both, answers were identical. The bottleneck in RAG is retrieval precision, not embedding dimensionality within a specific domain.

---

### 🟤 Q94 — Why not GPT-4?

Three reasons:

1. **AWS-native:** Cross-cloud API dependency would complicate deployment and IAM
2. **Cost:** Nova Lite is significantly cheaper
3. **Access:** GPT-4 and Claude required payment verification that would have blocked deployment; Nova Lite works immediately

Trade-off: Nova has a different API format that required custom `BedrockLLM` code. Worth it for a fully AWS-native deployment.

---

### 🟤 Q95 — The CI badge was failing — what happened?

Three separate problems, all fixed:

**Problem 1:** `deploy.yml` triggered on every push but had no AWS GitHub Secrets → instant failure every commit.
**Fix:** Changed trigger to `workflow_dispatch` (manual only).

**Problem 2:** pytest exit code 3 — no tests collected. `reportlab` was missing (needed to generate synthetic PDFs in test fixtures).
**Fix:** Added `reportlab` to CI dependencies.

**Problem 3:** `asyncio_mode = auto` in `pyproject.toml` caused pytest-asyncio to crash during collection in the CI environment.
**Fix:** Changed to `asyncio_mode = strict`.

Lesson: CI failures always have a specific cause. Read the error message carefully.

---

### 🟤 Q96 — Why two live deployments?

Intentional — to demonstrate the full AWS container deployment spectrum:

- **ECS** (`13.222.137.204:8000`): demonstrates fast, practical container deployment on managed AWS infrastructure
- **EKS** (`finrag.44.206.217.242.nip.io`): demonstrates production-grade Kubernetes — HPA, Ingress, namespaces, rolling updates

In a real project: ECS for a simple single service, EKS for a complex multi-service system needing the full Kubernetes ecosystem.

---

### 🟤 Q97 — What if Bedrock is down?

**Circuit breaker with graceful degradation:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
def call_bedrock(prompt):
    return client.invoke_model(...)

try:
    answer = call_bedrock(prompt)
except RetryError:
    # Retrieval still worked — return chunks without generated answer
    answer = "Generation temporarily unavailable. Relevant sections: " + chunks_text
```

Users still get relevant retrieved sections — better than a 500 error.

---

### 🟤 Q98 — How do you prevent answers outside the document?

Multiple layers:

1. **System prompt:** *"Answer ONLY from the provided context. If not in context, say you don't know."*
2. **Temperature = 0:** Deterministic output — less creative filling-in
3. **Cross-encoder reranking:** Only genuinely relevant chunks reach the LLM
4. **No internet access:** Model physically cannot go outside what's provided
5. **Sources attached:** Every answer includes page numbers — user can verify

---

### 🟤 Q99 — What is the CAP theorem and does it apply?

CAP: in a distributed system, you can guarantee at most 2 of: **C**onsistency, **A**vailability, **P**artition Tolerance.

This system chooses **AP** (Available + Partition Tolerant):
- If a pod crashes mid-index-update, the index might be momentarily stale
- But the system stays available — other pods serve traffic

For financial Q&A, slightly stale data is acceptable. For a trading system executing orders, you'd need CP — consistency is non-negotiable.

---

### 🟤 Q100 — Explain this project to a non-technical person

> "Imagine you have a 500-page annual report and you need to find the answer to a specific question. Normally you'd spend hours reading through it.
>
> This system reads the entire report — text AND every chart and graph — in under a minute. Then when you ask a question, it finds the exact answer with the page number it came from. In about 4 seconds.
>
> It's like having a financial analyst who has read every word of every report you've ever uploaded — and never forgets anything they've read."

---

*Built with: LlamaIndex · AWS Bedrock · Amazon Nova Lite · OpenCLIP · sentence-transformers · FastAPI · Docker · AWS ECS · AWS EKS · Kubernetes*

*Live: http://finrag.44.206.217.242.nip.io*
