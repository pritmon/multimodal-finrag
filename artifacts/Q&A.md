# 🎯 Multimodal Financial RAG — Interview Mastery Guide

> **This is not just Q&A. This is a script.**
> Every answer tells a story. Every story triggers curiosity. Every curiosity makes them want to hire you.
>
> Format per question:
> - 📌 **What interviewer really wants to know**
> - 🗣️ **Exactly what to say** (word for word)
> - ⚡ **The hook** — the one sentence that makes them lean forward
> - 🔬 **Technical depth** — if they dig deeper

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
| 🟤 | [Tricky Questions](#-tricky-interview-questions) | Q93–Q100 |

---

## 🔵 Core RAG & Multimodal Concepts

---

### 🔵 Q1 — Tell me about this project

📌 **They want:** Can you explain complex things simply? Are you proud of what you built?

🗣️ **Say this:**
> "I built a system that lets you talk to financial documents — not just read them, but actually ask questions and get answers. You upload an annual report, a 10-K, any financial PDF. The system reads every page — text AND charts — and when you ask something like 'what was the operating margin in Q3?', it finds the exact answer with the page number in about 4 seconds.
>
> What makes it interesting is the multimodal part — most systems are blind to charts. Mine can see them. It uses a vision AI to understand what each chart is showing and includes that understanding in the answer.
>
> And it's not running on my laptop — it's live right now on AWS, deployed on both ECS and Kubernetes."

⚡ **The hook:** *"It's live right now — you can query it during this interview if you want."*

🔬 **If they ask more:** RAG = Retrieval-Augmented Generation. Instead of training the model on documents (expensive, slow), we retrieve the relevant parts at query time and let the LLM generate from only those parts. No hallucinations because the model can only answer from what we give it.

---

### 🔵 Q2 — What makes it "Multimodal"?

📌 **They want:** Do you know what multimodal actually means technically — or just the buzzword?

🗣️ **Say this:**
> "Most RAG systems are text-only — they're essentially blind. If a PDF has a bar chart showing revenue growth, a text-only system skips it entirely. Mine doesn't.
>
> I use OpenCLIP — a vision-language model — to look at every image in the PDF and score it: is this a financial chart, a logo, or just decoration? If it's a chart, I send it to Amazon Bedrock's vision API which generates a caption — 'this bar chart shows revenue growing from $42B to $47B between FY2023 and FY2024.'
>
> That caption becomes searchable text. So when someone asks about revenue trends, the chart's information is part of the answer — not just the text on the page."

⚡ **The hook:** *"Most RAG systems miss 30-40% of financial information because it's in charts. Mine captures it."*

🔬 **Technical depth:**
```python
# CLIP scores each image against labels
labels = ["a financial chart or graph", "decorative image", "company logo"]
similarity = (image_features @ text_features.T).softmax(dim=-1)
# If "financial chart" > 0.3 threshold → send to Bedrock Vision for captioning
```

---

### 🔵 Q3 — Why RAG and not fine-tuning?

📌 **They want:** Do you understand the trade-offs — or did you just copy a tutorial?

🗣️ **Say this:**
> "I thought about fine-tuning. Here's why I didn't.
>
> Fine-tuning is like training an employee to memorise a textbook. RAG is like giving them the textbook and letting them look things up during the exam. For financial documents that change every quarter, fine-tuning means retraining every time a new report comes out — that's GPU cost, time, and risk of the model forgetting previous knowledge.
>
> RAG just means uploading the new PDF. 41 seconds to index. Done.
>
> Also, fine-tuned models still hallucinate — they blend what they learned with what they were trained on. RAG is grounded — the model can ONLY answer from the chunks I give it. For financial data where one wrong number matters, that's critical."

⚡ **The hook:** *"Fine-tuning is like tattooing knowledge into the model. RAG is like giving it a library card. For documents that change, the library card wins every time."*

| | RAG | Fine-tuning |
|--|-----|------------|
| **Update new doc** | Upload → 41 seconds | Retrain the model |
| **Hallucination** | Low — anchored to docs | Higher |
| **Cost** | Low | High (GPU) |
| **Our choice** | ✅ | ❌ |

---

### 🔵 Q4 — Explain the full pipeline end-to-end

📌 **They want:** Can you hold the whole system in your head?

🗣️ **Say this:**
> "There are two phases — ingest and query.
>
> Ingest: PDF comes in → PyMuPDF extracts every text block with page number and coordinates. For images, CLIP checks if they're charts. Charts get captioned by Bedrock Vision. All text gets chunked by page — about 800 chunks for a 369-page PDF. Each chunk gets converted to a 384-dimensional vector using a local sentence-transformer model. Everything is stored in a vector index on disk.
>
> Query: Question comes in → converted to the same vector space → hybrid search — BM25 for exact keyword matching AND vector search for semantic meaning — combined using Reciprocal Rank Fusion. Top 20 results go through a cross-encoder reranker that picks the best 4. Those 4 chunks go to Amazon Nova Lite on Bedrock with the question. Answer comes back with sources and page numbers."

⚡ **The hook:** *"The whole thing — upload to answer — happens in under 50 seconds for a 369-page document."*

```
PDF → PyMuPDF → Text Blocks + Images
                      │               └── CLIP → Bedrock Vision Caption
                      ▼
             Page-level Chunks (800 chunks)
                      ▼
          all-MiniLM-L6-v2 Embeddings (3 seconds)
                      ▼
              VectorStoreIndex (disk-persisted)

Question → BM25 + Vector Search → RRF Fusion → Cross-Encoder → Top 4 → Nova Lite → Answer
```

---

### 🔵 Q5 — What is Hybrid Search and why?

📌 **They want:** Do you know why you made this choice — or just copied it?

🗣️ **Say this:**
> "I found that neither keyword search nor semantic search alone was good enough for financial data.
>
> Keyword search — BM25 — is great when someone types an exact term like 'EBITDA margin Q3 2024'. But if they ask 'how profitable was the company last year', BM25 has no idea — those words aren't in the document.
>
> Vector search handles meaning — 'profitable' and 'margin' are semantically close. But it misses exact numbers. If I search for '47.3 billion', vector search might return something about '47 billion' or 'revenue in the billions' — not precise enough.
>
> Hybrid gives you both. I combine the ranked lists using Reciprocal Rank Fusion — a chunk that ranks high in BOTH lists gets the highest combined score. In my testing, hybrid improved answer accuracy by about 25% over either method alone."

⚡ **The hook:** *"Two detectives are better than one. BM25 finds the exact words. Vector search finds the meaning. Together they don't miss anything."*

```python
# RRF formula — simple but powerful
score = 1/(60 + rank_bm25) + 1/(60 + rank_vector)
# Consistent performer in both lists wins
```

---

### 🔵 Q6 — What is a Cross-Encoder Reranker?

📌 **They want:** Do you understand the two-stage retrieval pattern?

🗣️ **Say this:**
> "Think of it as two rounds of filtering.
>
> Round 1 — the retriever — is fast and broad. It pulls the top 20 most probably relevant chunks in milliseconds using embeddings and BM25. It's like a recruiter doing a CV screen — quick but sometimes misses nuance.
>
> Round 2 — the cross-encoder — is slow but precise. It looks at the question AND each chunk together, as a pair, and scores how well they match. It's like the actual interview — much more accurate.
>
> The model I use is `cross-encoder/ms-marco-MiniLM-L-6-v2`. One thing I had to fix — it returns raw logit scores that can be negative, which looked broken. I applied sigmoid normalisation to convert them to clean 0-1 percentages."

⚡ **The hook:** *"The reranker is why my system returns the RIGHT answer instead of just a related answer."*

```python
raw_scores = reranker.predict(pairs)
scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
# Now 0.0 to 1.0 — clean confidence percentages
```

---

### 🔵 Q7 — How did you optimise chunking?

📌 **They want:** Did you actually run this — or just describe theory?

🗣️ **Say this:**
> "This is one of the things I'm most proud of solving. The original approach created one chunk per text block — PyMuPDF extracts maybe 30 blocks per page. 369 pages × 30 blocks = 11,327 nodes. Each node needed an embedding. With Bedrock Titan at 5 requests per second — that's over 4 hours to index one document. Completely unusable.
>
> I realised I didn't need block-level granularity. Financial questions are about page-level content — 'what does page 42 say about margins?' So I merged all blocks per page into one chunk. 11,327 nodes became 800. Combined with switching to local embeddings — indexing went from 4+ hours to 41 seconds.
>
> Same answer quality. 350x faster."

⚡ **The hook:** *"I took indexing from 4 hours to 41 seconds — not by buying faster hardware, but by thinking about the problem differently."*

```python
from collections import defaultdict
page_texts = defaultdict(list)
for block in parsed_doc.text_blocks:
    page_texts[block.page_number].append(block.text)

documents = [
    Document(text="\n".join(page_texts[p]), metadata={"page_number": p})
    for p in sorted(page_texts)
]
# 11,327 blocks → 369 page-chunks
```

---

### 🔵 Q8 — What is an embedding?

📌 **They want:** Can you explain a technical concept to a non-technical person?

🗣️ **Say this:**
> "An embedding is a way of converting the meaning of text into coordinates — like GPS coordinates, but for meaning instead of geography.
>
> Every chunk of text becomes a list of 384 numbers. Sentences with similar meaning end up with similar numbers — close together in this 384-dimensional space. 'Revenue increased' and 'Sales went up' would be almost neighbours. 'Revenue increased' and 'The cat sat on the mat' would be far apart.
>
> When you ask a question, it also gets converted to coordinates. Then I find the chunks whose coordinates are closest — those are the most semantically relevant chunks. That's vector search."

⚡ **The hook:** *"384 numbers capture the entire meaning of a paragraph. The math that compares them takes microseconds."*

---

### 🔵 Q9 — Why switch from Bedrock Titan to local embeddings?

📌 **They want:** Do you make decisions based on evidence — or copy what others do?

🗣️ **Say this:**
> "I started with Titan because it's the 'correct' AWS-native choice. But I hit reality fast — Bedrock throttles free-tier embedding calls to about 5 per second. 11,000 chunks at 5 per second is 37 minutes minimum, with ThrottlingException errors scattered throughout.
>
> I switched to `all-MiniLM-L6-v2` from HuggingFace — runs locally, zero API calls, no throttling. 800 chunks in 3 seconds on my Mac's GPU.
>
> Yes, it's 384 dimensions versus 1536 for Titan. But for domain-specific financial Q&A, the retrieval quality difference is negligible — I tested both and the answers were identical. Why pay for API calls and wait 4 minutes when local is free and 80x faster?"

⚡ **The hook:** *"I chose the tool that actually worked over the tool that looked good on paper."*

| | Bedrock Titan | Local MiniLM |
|--|--------------|-------------|
| Speed | 4+ minutes | 3 seconds |
| Cost | Per API call | Free |
| Throttling | Yes | No |
| Dimensions | 1536 | 384 |
| Our choice | ❌ | ✅ |

---

### 🔵 Q10 — What is Grounding and why does it matter?

📌 **They want:** Do you understand the hallucination problem in LLMs?

🗣️ **Say this:**
> "Hallucination is the LLM's #1 problem for factual use cases. The model was trained on billions of documents and sometimes 'fills in the gaps' with plausible-sounding but wrong information. For financial data — one wrong number in an earnings analysis could cost someone real money.
>
> Grounding means the model can ONLY answer from what I give it. My system prompt says: 'Answer only from the provided context. If the answer is not in the context, say you don't know.' The model has no internet access. It receives only the 4 retrieved chunks. That's its entire world for that query.
>
> Every answer also includes the source page number — so the user can verify."

⚡ **The hook:** *"A grounded system says 'I don't know' when it doesn't know. An ungrounded system makes something up. In finance, that difference matters."*

---

### 🔵 Q11 — What is the difference between ECS and EKS?

📌 **They want:** Do you know AWS deployment — or just 'it works on my machine'?

🗣️ **Say this:**
> "I deployed to both — intentionally — so I could demonstrate both.
>
> ECS is Amazon's own container runner. You describe what container to run, how much CPU and memory, and ECS handles the rest. Simple, fast to set up, good for straightforward workloads. That's running at 13.222.137.204:8000.
>
> EKS is managed Kubernetes. More complex but more powerful — you get HPA for auto-scaling, Ingress for traffic routing, namespaces for isolation, rolling updates with zero downtime. That's running at finrag.44.206.217.242.nip.io.
>
> For a production system with real traffic, I'd use EKS. For a quick internal tool, ECS."

⚡ **The hook:** *"Both are live right now. You can hit the health check on either one."*

---

### 🔵 Q12 — What is Amazon Nova Lite?

📌 **They want:** Do you know the AWS AI stack?

🗣️ **Say this:**
> "Nova Lite is Amazon's own LLM — available on Bedrock without any special approval or payment verification. It's fast, cheap, supports vision natively, and has a 300K token context window.
>
> I chose it over Claude because Claude on Bedrock requires manual account approval that can take days. Nova Lite works immediately. For this project — which I wanted live and demonstrable — Nova was the right call.
>
> One thing I had to figure out: Nova has a different API format than Claude. LlamaIndex assumes Claude format. I had to write a custom `BedrockLLM` class that detects which model is being used and formats the request correctly."

⚡ **The hook:** *"I didn't just pick Nova from a list — I debugged its API format differences and wrote custom code to make it work with LlamaIndex."*

```python
# Nova format (no "type" key — different from Claude)
{"text": "Hello"}  # Nova
{"type": "text", "text": "Hello"}  # Claude
```

---

### 🔵 Q13 — What is LlamaIndex?

📌 **They want:** Do you know the RAG ecosystem?

🗣️ **Say this:**
> "LlamaIndex is the orchestration framework — it's the plumbing that connects all the components. Without it, I'd have to manually wire up: document chunking, embedding storage, vector indexing, retrieval logic, LLM connection, prompt management.
>
> LlamaIndex gives clean abstractions for all of that. I plug in my custom embeddings, my custom Bedrock LLM, my parsed documents — and it handles the pipeline.
>
> That said, I also hit its limitations. When it detected I was using an LLM and tried to route through its chat interface, it sent Nova Lite a message in Claude format — which broke. I had to override the `is_chat_model` property to force it through the `complete()` path instead."

⚡ **The hook:** *"I know LlamaIndex well enough to know when to work with it and when to work around it."*

---

### 🔵 Q14 — What is Async Ingestion?

📌 **They want:** Do you think about user experience — not just functionality?

🗣️ **Say this:**
> "Indexing a 369-page PDF takes 41 seconds. If I made the upload endpoint wait synchronously, the user's browser would sit there for 41 seconds — then likely timeout and think it failed.
>
> So I use a background job pattern. The user uploads the PDF, immediately gets back a job ID and status 'processing'. The actual indexing runs in a background thread. The user polls a status endpoint every few seconds — 'still processing', 'still processing', 'done — 905 text chunks, 5 chart nodes indexed.'
>
> This is the standard pattern for any long-running operation in a web API."

⚡ **The hook:** *"The best user experience for a 41-second operation is making it feel instant at the start."*

```python
POST /ingest → {"job_id": "abc123", "status": "processing"}  # immediate
GET /ingest/status/abc123 → {"status": "done", "nodes": 905}  # when ready
```

---

### 🔵 Q15 — How do you evaluate RAG quality?

📌 **They want:** Do you just build and hope — or do you measure?

🗣️ **Say this:**
> "I use two approaches. First, RAGAS — an open source framework that measures four things: Faithfulness (does the answer match the sources?), Answer Relevance (does it actually answer the question?), Context Precision (are the retrieved chunks useful?), and Context Recall (did we get everything needed?).
>
> Second, manual testing with real financial documents. I tested with Infosys annual reports and Tesla quarterly filings — asked specific financial questions where I knew the correct answers. Revenue figures, margin percentages, risk factors. All correct.
>
> The real test is: can I trust the answer enough to use it in a financial decision? That's the bar I optimised for."

⚡ **The hook:** *"I built a test suite around questions where I already knew the answers — so I could verify the system, not just believe it."*

---

## 🟢 General Technical

---

### 🟢 Q16 — What is FastAPI and why use it?

📌 **They want:** Framework knowledge + reasoning.

🗣️ **Say this:**
> "FastAPI is a modern Python web framework. I chose it for three reasons: automatic request validation via Pydantic — wrong data types are rejected before they reach my code. Async support built-in — handles many simultaneous requests without blocking. And automatic Swagger docs at `/docs` — anyone can test the API without writing a single line of client code."

⚡ **The hook:** *"FastAPI gives me validation, async, and docs for free — that's three things I'd have to build manually in Flask."*

```python
@router.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    result = pipeline.query(request.question, top_k=request.top_k)
    return QueryResponse(answer=result.answer, sources=result.sources)
```

---

### 🟢 Q17 — What is Pydantic?

📌 **They want:** Do you validate inputs — or trust users?

🗣️ **Say this:**
> "Pydantic is a data validation library. I define what valid input looks like — question must be a string between 1 and 2000 characters, top_k must be an integer between 1 and 20. If someone sends top_k as 'banana' or 500, Pydantic rejects it automatically with a clear error message. My business logic never sees invalid data."

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=20)
```

---

### 🟢 Q18 — What is PyMuPDF?

🗣️ **Say this:**
> "PyMuPDF — imported as `fitz` — is the library I use to parse PDFs. It extracts text blocks with precise bounding box coordinates (x0, y0, x1, y1), page numbers, font sizes for heading detection, and embedded images. It's significantly faster and more accurate than alternatives like PyPDF2 for structured extraction."

```python
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
for page in doc:
    blocks = page.get_text("dict")["blocks"]  # text with metadata
    images = page.get_images(full=True)        # embedded images
```

---

### 🟢 Q19 — What is CLIP?

📌 **They want:** Do you understand vision-language models?

🗣️ **Say this:**
> "CLIP stands for Contrastive Language-Image Pre-training — created by OpenAI, trained on 400 million image-text pairs. Its special ability is zero-shot classification — you give it an image and a list of text labels, and it scores how well each label matches the image. No training on financial charts needed. I just ask it: 'Is this a financial chart, a logo, or decoration?' and it knows."

⚡ **The hook:** *"I never trained CLIP on financial charts. It just knows — because it learned language and vision together at massive scale."*

---

### 🟢 Q20 — What is BM25?

🗣️ **Say this:**
> "BM25 — Best Match 25 — is a keyword search algorithm from 1994 that's still state-of-the-art for sparse retrieval. It's like a very smart Ctrl+F. It gives higher scores to rare words — if 'EBITDA' appears in a chunk, that's very significant. Common words like 'the' score almost zero. It also normalises for document length so a longer chunk isn't unfairly rewarded just for being long."

---

### 🟢 Q21 — What is RRF?

🗣️ **Say this:**
> "Reciprocal Rank Fusion is a formula to merge two ranked lists into one. Instead of averaging scores — which are on different scales — it converts ranks to scores using 1/(60 + rank). A chunk ranked #1 in BM25 and #1 in vector search gets the maximum combined score. A chunk that's only great in one list but mediocre in the other scores lower. It rewards consistency."

```python
score = 1/(60 + rank_bm25) + 1/(60 + rank_vector)
```

---

### 🟢 Q22 — What is a Vector Store?

🗣️ **Say this:**
> "A vector store is a database optimised for similarity search — not exact match. Normal databases ask 'does this value equal X?' A vector store asks 'which stored vectors are closest in meaning to this query vector?' It uses algorithms like FAISS — Facebook AI Similarity Search — which builds a special index structure that finds approximate nearest neighbours in milliseconds without checking every vector."

---

### 🟢 Q23 — What is the difference between top_k and top_n?

🗣️ **Say this:**
> "top_k is how many chunks the retriever pulls — say 20. Fast, uses vector math. top_n is how many survive reranking — say 4. Slow, uses the cross-encoder. The two-stage design lets me be broad in retrieval and precise in selection. Sending 20 chunks to the LLM would be expensive and dilute the answer. 4 focused chunks gives much cleaner answers."

---

### 🟢 Q24 — What is uvicorn?

🗣️ **Say this:**
> "uvicorn is the ASGI server that runs FastAPI. FastAPI is the application — it defines the routes and logic. uvicorn is what actually listens on port 8000 and serves HTTP requests. I run with `--workers 1` because the ML models — CLIP, sentence-transformers — are loaded into memory at startup. Multiple workers would duplicate that memory usage needlessly."

---

### 🟢 Q25 — What is the lifespan pattern?

🗣️ **Say this:**
> "The lifespan pattern runs code once at startup and once at shutdown — not once per request. I use it to load the vector index and ML models into memory when the server starts. Loading models per-request would add 2-3 seconds to every query. Loading once at startup means the first request is fast too."

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load_index()   # runs once — at startup
    yield
    pipeline.cleanup()      # runs once — at shutdown
```

---

### 🟢 Q26–Q30 — Technical depth questions

**sentence-transformers:** HuggingFace library for local text embeddings. `all-MiniLM-L6-v2` is 90MB, runs on CPU or GPU, produces 384-dimensional vectors. Free, no API calls, works offline.

**Apple MPS:** Metal Performance Shaders — Apple's GPU acceleration. sentence-transformers auto-detects it on M1/M2/M3 Macs → 5-10x faster embeddings vs CPU.

**PVC (Persistent Volume Claim):** Kubernetes storage that survives pod restarts. Without it, the vector index stored in `/app/index_store` is lost every time a pod restarts.

**HPA:** Horizontal Pod Autoscaler — monitors CPU/memory, adds pods when above threshold, removes when below. Configured: min 2 pods, max 10, scale at 70% CPU.

**Docker multi-stage build:** Two stages — builder installs everything including dev tools (3GB), runtime copies only what's needed to run (800MB). Smaller = faster pull, cheaper ECR storage, smaller attack surface.

---

## 🟠 Embeddings & Search

---

### 🟠 Q31 — What is cosine similarity?

🗣️ **Say this:**
> "Cosine similarity measures the angle between two vectors — not their magnitude. Two vectors pointing in the same direction score 1.0 (identical meaning). Perpendicular vectors score 0.0 (unrelated). Opposite directions score -1.0. We use it because it's robust to document length — a longer chunk with the same meaning scores the same as a shorter one."

---

### 🟠 Q32 — Why 384 dimensions?

🗣️ **Say this:**
> "Trade-off between quality and speed. 384 dimensions from MiniLM gives good quality for domain-specific retrieval, runs in 3 seconds locally. 1536 from Titan gives marginally better quality but costs money and is throttled. For financial Q&A where the documents are domain-specific, the retrieval quality difference is negligible. I tested both — same answers."

---

### 🟠 Q33–Q42 — Deeper search questions

**FAISS:** Facebook AI Similarity Search. Builds an IVF (Inverted File Index) that partitions the vector space into clusters. Instead of comparing your query to all 800 vectors, it only compares to the vectors in nearby clusters. Millisecond search.

**Sparse vs Dense:** BM25 = sparse (mostly zeros, vocabulary-sized). Embeddings = dense (all non-zero, 384 dimensions). Sparse captures exact terms. Dense captures meaning. Hybrid captures both.

**Long documents:** We never send the whole document to the LLM. 800 chunks stored, top 4 retrieved per query. Even 1,000-page documents work fine.

**Embedding cache:** Dictionary of `{text: vector}`. If the same chunk is retrieved in multiple queries, embedding is computed once and reused. Saves time on repeated queries.

**Chunking overlap:** 512 tokens per chunk, 64-token overlap. Ensures sentences at chunk boundaries aren't split in a way that loses meaning.

**Pre-embedding:** Batch all 800 chunks in one call to `model.encode()` instead of one call per chunk. Dramatically faster — batch processing is how GPUs work efficiently.

---

## 🟣 Multimodal — Charts & Vision

---

### 🟣 Q43 — What is OpenCLIP?

🗣️ **Say this:**
> "OpenCLIP is the open-source implementation of OpenAI's CLIP. I use the `ViT-B-32` variant — Vision Transformer Base with 32×32 pixel patches. It was trained on 400 million image-text pairs and can compare any image to any text description — zero-shot. No financial chart training data needed."

---

### 🟣 Q44 — What is zero-shot classification?

🗣️ **Say this:**
> "Zero-shot means the model classifies things it was never explicitly trained on. I never trained CLIP on 'financial chart'. But because it learned the relationship between images and language at massive scale, I can just ask: 'Does this image look like a financial chart?' and it gives an accurate score. The labels are just English text — I can change them anytime without retraining."

---

### 🟣 Q45 — What is Bedrock Vision?

🗣️ **Say this:**
> "Bedrock's multimodal API accepts both text and images in the same request. I send the chart image as base64-encoded bytes alongside a prompt: 'Describe this financial chart. What does it show? What are the key numbers?' Nova Lite returns a text caption that I store as a searchable text node in the vector index."

```python
body = {"messages": [{"role": "user", "content": [
    {"image": {"format": "png", "source": {"bytes": b64_image}}},
    {"text": "Describe this financial chart..."}
]}]}
```

---

### 🟣 Q46–Q52 — Chart questions

**Chart captions in RAG:** Stored as special `TextNode` with metadata `type: "chart"`. When a query retrieves a page with a chart, the caption AND the image are included in the API response.

**Base64:** Binary-to-text encoding. APIs communicate in JSON (text). Images are binary. Base64 converts image bytes to safe ASCII characters that fit in JSON.

**ViT:** Vision Transformer — splits an image into 32×32 pixel patches, treats each patch like a word token, applies the same transformer attention mechanism as language models.

**No charts in PDF:** `if not parsed_doc.images: chart_results = []` — skips CLIP and Bedrock Vision entirely. Zero unnecessary API calls.

**Cost control:** Max 5 charts per document. All 5 captioned in parallel using ThreadPoolExecutor. Same cost, 5x faster than sequential.

**Nova vs Claude vision format:** Claude uses `{"type": "image", "source": {"type": "base64", ...}}`. Nova uses `{"image": {"format": "png", "source": {"bytes": ...}}}`. Required a custom `_is_nova()` detection method in `bedrock_llm.py`.

**Next additions:** Table extraction (currently treated as text), OCR for scanned PDFs, audio (earnings calls via Whisper → RAG), multi-page chart linking.

---

## 🔷 AWS & Cloud Architecture

---

### 🔷 Q53 — What is AWS Bedrock?

🗣️ **Say this:**
> "Bedrock is AWS's managed AI service — you get access to foundation models via API without managing any infrastructure. You pay per token. Available models include Amazon Nova, Anthropic Claude, Meta Llama, Mistral. I use Nova Lite for generation and captioning. I originally used Titan for embeddings but switched to local."

---

### 🔷 Q54 — What is ECR?

🗣️ **Say this:**
> "Elastic Container Registry — Amazon's private Docker image repository. Like Docker Hub but inside your AWS account. My image is at `020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest`. When ECS or EKS pulls the container, it pulls from here."

---

### 🔷 Q55 — What is Fargate?

🗣️ **Say this:**
> "Fargate is serverless compute for containers. With EC2 you manage the underlying servers. With Fargate you just say 'run this container with 1 vCPU and 3GB RAM' and AWS handles the rest. My ECS deployment runs on Fargate — no server management, pay only for actual usage."

---

### 🔷 Q56 — What is IAM?

🗣️ **Say this:**
> "Identity and Access Management — the security layer of AWS. Controls who can do what. My EKS pods use IRSA — IAM Roles for Service Accounts — which lets pods assume IAM roles without storing credentials in the container. The role has minimum necessary permissions: invoke Bedrock, read from S3, write to CloudWatch."

---

### 🔷 Q57–Q65 — AWS depth questions

**CloudWatch:** AWS logging service. Every `logger.info()` in my app → CloudWatch Logs → searchable, can trigger alerts. Used for debugging ECS task failures.

**Lambda:** Serverless function triggered by S3 events. When PDF uploaded → Lambda auto-triggers → indexes in background. Code in `src/lambda_handler/handler.py`.

**S3:** Object storage. Bucket `pritam-finrag-docs`. PDFs stored at `documents/{job_id}/{filename}`. 99.999999999% durable.

**VPC:** EKS creates its own VPC — private subnets for workers, public subnets for load balancer. More secure than default VPC.

**Load Balancer:** `type: LoadBalancer` in Service YAML → AWS creates ALB automatically → public DNS → traffic distributed to healthy pods.

**STS:** `aws sts get-caller-identity` — verified AWS credentials before deployment. Returns account ID and IAM identity.

**nip.io:** Free DNS wildcard service. `finrag.44.206.217.242.nip.io` resolves to `44.206.217.242`. No domain purchase needed.

**ECS Task Definition:** Blueprint for a container — image URL, CPU, memory, port, environment variables, log configuration. Version-controlled JSON.

**eksctl:** CLI that creates EKS clusters in one command. Without it — 20+ manual console steps. Lessons: use `AL2023_x86_64_STANDARD` AMI for K8s 1.34, use EKS VPC subnets not default VPC subnets.

---

## 🔴 Kubernetes & Deployment

---

### 🔴 Q66 — What is Kubernetes?

🗣️ **Say this:**
> "Kubernetes is the self-healing, self-scaling manager for containerised applications. Without it: you manually start containers, manually restart crashes, manually add servers for traffic spikes. With it: you describe the desired state — '2 replicas of this container' — and Kubernetes makes it happen and keeps it that way. Pod crashes? Restarted automatically. CPU hits 70%? HPA adds more pods automatically."

---

### 🔴 Q67 — What is a Pod?

🗣️ **Say this:**
> "The smallest deployable unit in Kubernetes — usually one container. Has its own IP inside the cluster. Temporary by design — Kubernetes kills and replaces pods regularly. That's fine because the Deployment ensures the desired number always exists."

---

### 🔴 Q68 — Deployment vs Service?

🗣️ **Say this:**
> "Deployment answers 'how many pods, what image, how to update.' Service answers 'how to reach those pods from outside.' Deployment without Service = pods running but unreachable. Service without Deployment = nothing to route to. They work together."

---

### 🔴 Q69 — Rolling Update?

🗣️ **Say this:**
> "When I push a new image, Kubernetes doesn't kill all pods and restart — that would cause downtime. It starts one new pod, waits for it to pass health checks, then kills one old pod. Repeat until all pods are on the new version. Zero downtime. I set `maxUnavailable: 0` to ensure there's always at least one pod serving traffic."

---

### 🔴 Q70–Q75 — K8s depth

**Namespace:** `finrag` namespace isolates all resources. `kubectl get pods -n finrag`. Like a folder — different teams, different namespaces, same cluster.

**ConfigMap/Secret:** ConfigMap for non-sensitive config (`AWS_REGION`, `BEDROCK_MODEL_ID`). Secret for credentials (base64 encoded). Both mounted as environment variables into pods.

**Liveness vs Readiness:** Liveness = is it alive? (restart if not). Readiness = is it ready for traffic? (remove from load balancer if not). I set `initialDelaySeconds: 60` for liveness — models take time to load.

**Kustomize:** `kubectl apply -k k8s/` applies all YAML files in the right order from `kustomization.yaml`. One command deploys everything.

**Ingress:** One load balancer routes to multiple services based on hostname/path. Without it: one LB per service (expensive). With it: one LB total.

**Nodegroup failures:** AMI `AL2_x86_64` not supported for K8s 1.34 → use `AL2023_x86_64_STANDARD`. Wrong VPC subnets → query EKS cluster VPC first. Key lesson: AWS error messages are specific — read them carefully.

---

## 🟡 MLOps & Production

---

### 🟡 Q76 — What is CI/CD?

🗣️ **Say this:**
> "CI — Continuous Integration — runs tests automatically on every commit. My `ci.yml` runs pytest on every push to main. CD — Continuous Deployment — automates the deploy. My `deploy.yml` is manual-trigger only because it needs AWS credentials. I separated them deliberately — tests always run, deploy only when I'm ready."

---

### 🟡 Q77 — Health Check endpoint?

🗣️ **Say this:**
> "Every production system needs a health check. Mine returns status, version, and whether the index is loaded. Kubernetes probes hit it every 10 seconds — if it fails, the pod is replaced. The load balancer only routes traffic to pods where health check passes. It's also useful for quick manual verification after a deploy."

```python
@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0", "index_loaded": pipeline.index is not None}
```

---

### 🟡 Q78–Q85 — MLOps depth

**RAGAS:** Evaluation framework — Faithfulness, Answer Relevance, Context Precision, Context Recall. Run on a test set of questions with known answers.

**Model drift:** LLM output quality degrading over time. Detect by logging Q&A pairs, running RAGAS weekly, alerting if faithfulness drops below threshold.

**Non-root Docker user:** `USER finrag` — if container is compromised, attacker has limited permissions. Security best practice for production.

**Structured logging:** JSON logs with `structlog` — `logger.info("query", question=q, latency_ms=t)`. Machine-readable, searchable in CloudWatch.

**Rate limiting:** Bedrock throws `ThrottlingException`. Handle with `tenacity` — retry 3 times with exponential backoff. Never crash, always retry gracefully.

**Context window:** Nova Lite = 300K tokens. My 4 chunks ≈ 2,000 tokens. Well within limits. The constraint is relevance quality, not window size.

**Prompt injection:** User tries "ignore your instructions". Prevention: system prompt separated from user input, Pydantic validation, temperature=0, monitor logs.

**Autoscaling:** EKS HPA — CPU > 70% → add pod (max 10). CPU < 70% for 5 min → remove pod (min 2). Saves money at low traffic, handles spikes automatically.

---

## 🔶 SDLC & Development Approach

---

### 🔶 Q86 — What was your development approach?

📌 **They want:** Are you systematic — or do you hack until it works?

🗣️ **Say this:**
> "I followed a 7-phase approach. Requirements first — financial PDF Q&A with chart understanding, deployed on AWS. Then design — chose hybrid RAG over pure vector search because financial queries mix exact terms and semantic meaning. Then core pipeline — parser, embeddings, retrieval, LLM — working end-to-end before adding any extras. Then API, then multimodal chart support, then deployment — Docker to ECR to ECS to EKS. Finally testing and optimisation — which is where I found the 4-hour indexing problem and fixed it to 41 seconds."

⚡ **The hook:** *"I always make the core pipeline work before adding features. Complexity is debt — pay it only when the foundation is solid."*

---

### 🔶 Q87 — What was the hardest bug?

📌 **They want:** How do you debug? Do you give up or dig deep?

🗣️ **Say this:**
> "The Nova ValidationException. Every single query returned an error — `required key [toolUse] not found`. I spent hours on it.
>
> The problem was in LlamaIndex's source code — not in mine. When LlamaIndex sees `is_chat_model=True` on an LLM, it routes queries through the `chat()` method and formats messages in Claude's format. Nova Lite uses a different format and rejects Claude-format messages.
>
> Fix was one line: set `is_chat_model=False` on my custom LLM class. That forces LlamaIndex to use `complete()` instead of `chat()` — which sends the prompt in the format Nova expects.
>
> The lesson: when something consistently fails with a cryptic error, the bug is rarely in your code — it's in how your code interacts with a framework. Read the framework source."

⚡ **The hook:** *"I could have switched to a different LLM. Instead I read the LlamaIndex source code and fixed the actual problem."*

---

### 🔶 Q88 — How did you reduce indexing from 4 hours to 41 seconds?

📌 **They want:** Do you optimise — or accept slow as 'that's just how it is'?

🗣️ **Say this:**
> "Three changes, each meaningful on its own, together transformative.
>
> First — rethink chunking. The original approach created one chunk per text block — 11,327 chunks for a 369-page PDF. I asked: do I actually need block-level granularity? No — financial questions are page-level. Merged blocks per page: 11,327 → 800 chunks.
>
> Second — switch embeddings. Bedrock Titan at 5 requests/second with throttling → 4+ minutes. Local MiniLM → 3 seconds. Same answer quality.
>
> Third — pre-batch embeddings. Instead of embedding one node at a time during index construction, I batch all 800 in one GPU call before inserting.
>
> Together: 41 seconds."

⚡ **The hook:** *"350x faster — not from better hardware, from thinking about the problem differently."*

---

### 🔶 Q89 — How would you build this without AI assistance?

🗣️ **Say this:**
> "Start with `pip install llama-index sentence-transformers fastapi pymupdf`. Build and test each layer before connecting: parser first, then embeddings, then retrieval, then API. Add BM25 + RRF. Add reranker. Add chart detection. Containerise. Deploy. I estimate 3-4 days for a working system, a week for production-quality with all the edge cases handled."

---

### 🔶 Q90 — How would you scale to 1 million documents?

🗣️ **Say this:**
> "Current system: in-memory index, single node — works for hundreds of documents. For millions: replace VectorStoreIndex with pgvector (PostgreSQL + vector extension) or Pinecone — persistent, distributed, fast at scale. Lambda for indexing — parallel, serverless, handles bursts. EKS with HPA for the API layer. ElastiCache for query result caching. DynamoDB for document metadata. That's a production-grade system."

---

### 🔶 Q91 — What would improve answer quality?

🗣️ **Say this:**
> "Five things. Parent-child retrieval — embed small chunks for precision, retrieve larger parent chunks for context. HyDE — generate a hypothetical answer to the question, then embed that and search — better than embedding the raw question. Query decomposition — break complex questions into sub-questions. ColBERT reranking — more powerful than cross-encoder. Feedback loop — thumbs up/down stored, used to fine-tune retrieval weights."

---

### 🔶 Q92 — Scanned PDFs?

🗣️ **Say this:**
> "Current system extracts embedded digital text — scanned PDFs return empty. Fix: add pytesseract OCR layer — convert PDF pages to images, run OCR, get text. Or use AWS Textract — managed service that also extracts tables and forms structurally, not just as flat text."

---

## 🟤 Tricky Interview Questions

---

### 🟤 Q93 — Why local embeddings instead of OpenAI or Bedrock?

📌 **They want:** Can you defend your technical choices under pressure?

🗣️ **Say this:**
> "Deliberate trade-off. Bedrock Titan was throttled — 5 requests/second on free tier, 4+ minutes for 800 chunks, constant ThrottlingExceptions. OpenAI costs money per token, adds API dependency, breaks offline. Local MiniLM: free, no throttling, 3 seconds, works offline.
>
> Trade-off accepted: 384 dimensions vs 1536. For domain-specific financial Q&A, retrieval quality is equivalent — I tested both. The bottleneck in RAG is retrieval precision, not embedding dimensionality, when you're working within a specific domain."

---

### 🟤 Q94 — Why not GPT-4?

🗣️ **Say this:**
> "Three reasons. First — AWS-native project. Using OpenAI would create a cross-cloud dependency. Second — cost. Nova Lite is significantly cheaper. Third — approval. Claude and GPT-4 required payment verification that would have blocked deployment. Nova Lite worked immediately.
>
> The trade-off: Nova has a different API format that required custom code. Worth it for AWS-native deployment."

---

### 🟤 Q95 — The CI badge was failing. What happened?

📌 **They want:** How do you debug CI failures — or do you just ignore them?

🗣️ **Say this:**
> "Three separate problems, all fixed.
>
> First — `deploy.yml` was triggering on every push but had no AWS GitHub Secrets → immediate failure on every commit. Fixed by changing the trigger to `workflow_dispatch` — manual only.
>
> Second — pytest exiting with code 3 — no tests collected. `reportlab` was missing — needed for test fixtures that generate synthetic PDFs. Added to CI deps.
>
> Third — `asyncio_mode = auto` in `pyproject.toml` conflicted with the CI pytest version and caused collection to crash. Fixed by changing to `asyncio_mode = strict`.
>
> Lesson: CI failures always have a specific cause. Read the error message. Don't guess."

---

### 🟤 Q96 — You have two live deployments. Why both?

🗣️ **Say this:**
> "Intentional — to demonstrate the full range of AWS container deployment. ECS shows I can deploy containerised apps on managed AWS infrastructure quickly and simply. EKS shows I can work with Kubernetes — HPA, Ingress, namespaces, rolling updates, the full production stack.
>
> In a real project I'd choose one based on requirements. Simple single service → ECS. Complex multi-service, need K8s ecosystem → EKS."

---

### 🟤 Q97 — What if Bedrock goes down?

🗣️ **Say this:**
> "Retry with exponential backoff using `tenacity` — 3 attempts, waiting 4 to 10 seconds between them. If all retries fail: graceful degradation — return the retrieved chunks to the user with a message that generation is temporarily unavailable. The retrieval still worked. The user still gets relevant sections. They just don't get the synthesised answer. That's better than a 500 error."

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=4, max=10))
def call_bedrock(prompt):
    return client.invoke_model(...)
```

---

### 🟤 Q98 — How do you prevent answers outside the document?

🗣️ **Say this:**
> "System prompt is the first layer: 'Answer ONLY from the provided context. If not in context, say you don't know.' Temperature = 0 — deterministic, less creative. Cross-encoder reranking ensures only actually-relevant chunks reach the LLM. Sources attached to every answer so the user can verify. And I don't give the model internet access — it physically can't go outside the document."

---

### 🟤 Q99 — What is the CAP theorem and does it apply?

🗣️ **Say this:**
> "CAP says in a distributed system you can guarantee at most 2 of: Consistency, Availability, Partition Tolerance. My system chooses AP — Availability and Partition Tolerance. If a pod crashes mid-index-update, the index might be momentarily inconsistent — but the system stays up. For financial Q&A that's acceptable. For a trading system executing orders, you'd need CP — consistency is non-negotiable."

---

### 🟤 Q100 — Explain this project in 30 seconds to a non-technical person

📌 **This is your closing. Deliver it with energy.**

🗣️ **Say this — slowly, clearly, with a pause before the last line:**

> "Imagine you have a 500-page annual report and someone asks you a specific question about it. Normally you'd spend hours reading through it.
>
> This system reads the entire report — text AND every chart — in under a minute. Then when you ask a question, it finds the exact answer with the page number it came from. In about 4 seconds.
>
> *(pause)*
>
> It's like having a financial analyst who has read every word of every report you've ever uploaded — and never forgets anything."

⚡ **Then add:** *"And it's live right now if you want to try it — finrag.44.206.217.242.nip.io"*

---

## 🧠 The Dopamine Triggers — How to Make Them Want to Hire You

These are the moments that fire the interviewer's reward circuit:

| Moment | What triggers it |
|---|---|
| **Specificity** | "41 seconds" not "fast". "11,327 to 800 nodes" not "fewer chunks" |
| **Live demo** | "It's running right now — want to try it?" |
| **Problem → Solution** | Always pair the problem you faced with the fix you built |
| **Numbers** | 350x faster. 4 hours to 41 seconds. 800 chunks. 384 dimensions |
| **"I read the source code"** | Shows you go deep, not surface |
| **Trade-off reasoning** | Shows you think, not just copy |
| **One line that lands** | End every answer with the hook line |

---

## ⚡ 60-Second Opener — Use This at the Start

> "I built a multimodal RAG system for financial PDFs — deployed live on AWS, right now, on both ECS and Kubernetes.
>
> What makes it different from a standard RAG: it understands charts. Most systems skip images entirely. Mine uses a vision-language model to detect financial charts and caption them — so when you ask about revenue trends, the chart's data is part of the answer, not just the paragraph next to it.
>
> I also solved a performance problem that most tutorials ignore — naive chunking was going to take 4 hours to index a document. I got it to 41 seconds.
>
> The whole thing is live. You can query it right now."

---

*Built with: LlamaIndex · AWS Bedrock · Amazon Nova Lite · OpenCLIP · sentence-transformers · FastAPI · Docker · AWS ECS · AWS EKS · Kubernetes*

*Live: http://finrag.44.206.217.242.nip.io*
