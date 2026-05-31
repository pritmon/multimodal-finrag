# ☁️ AWS, Bedrock & Kubernetes — 
---

## 🗂️ Quick Navigation

| Colour | Section | Topics |
|--------|---------|--------|
| 🔵 | [AWS Fundamentals](#-aws-fundamentals) | What is AWS, S3, IAM, Regions, Free Tier |
| 🟣 | [AWS Bedrock & AI Models](#-aws-bedrock--ai-models) | Bedrock, Nova Lite, Titan Embeddings, Throttling |
| 🟢 | [AWS Compute & Serverless](#-aws-compute--serverless) | Lambda, boto3, ECS vs EKS |
| 🔴 | [Kubernetes Fundamentals](#-kubernetes-k8s-fundamentals) | Containers, Pods, Deployments, Services |
| 🟠 | [Kubernetes Advanced](#-kubernetes-advanced) | HPA, EKS, kubectl, Ingress |
| 🔷 | [Full Architecture](#-how-it-all-connects-in-this-project) | End-to-end flow, cheat sheet |

---

## 🔵 AWS Fundamentals

---

### 🔵 Q1 — What is AWS and why do companies use it?

💡 **Think of AWS like a rental shop for computers.**

Imagine you want to open a restaurant. You need:
- A kitchen (server/computer)
- A fridge (storage)
- A phone line (networking)
- A billing system

You could buy all this yourself — very expensive. Or you could **rent** it from someone who already has everything built.

**AWS = Amazon's rental shop for computers, storage, and internet tools.**

| Without AWS | With AWS |
|---|---|
| Buy servers upfront ($10,000+) | Pay per hour (cents) |
| Hire ops team to manage hardware | AWS manages it |
| Takes weeks to set up | Minutes to deploy |
| Fixed capacity | Scale up/down instantly |

Netflix, Airbnb, NASA — they all use AWS. In this project we use AWS for storage (S3), AI (Bedrock), containers (EKS/ECS), and serverless (Lambda).

---

### 🔵 Q2 — What is the AWS Free Tier?

💡 **Think of it as a 12-month free trial.**

When you create an AWS account, Amazon gives you free usage for 12 months on many services.

| Service | Free Tier Limit |
|---|---|
| S3 | 5 GB storage + 20,000 GET requests/month |
| Bedrock | Limited model invocations per month |
| EC2 | 750 hours/month of `t2.micro` instance |
| Lambda | 1 million requests/month free forever |

> ⚠️ **Lesson learned:** We hit throttling during bulk PDF indexing — 800 chunks × Titan Embeddings exceeded the free-tier rate limit. Fix: batch with delays.

---

### 🔵 Q3 — What is S3 and how is it used in this project?

💡 **Think of S3 as Google Drive for developers.**

**S3 = Simple Storage Service.** You store files (called "objects") in "buckets" (like folders).

In this project:
- **Bucket:** `pritam-finrag-docs`
- **Purpose:** Store uploaded financial PDFs before indexing
- **Key format:** `documents/{job_id}/infosys_annual_report_2025.pdf`

```python
# From src/ingestion/s3_loader.py
s3.put_object(
    Bucket="pritam-finrag-docs",
    Key="documents/abc123/infosys.pdf",
    Body=pdf_bytes
)
```

| Local Disk | S3 |
|---|---|
| Lost if server crashes | 99.999999999% durable |
| Not shared between pods | Accessible from anywhere |
| Fixed size | Unlimited storage |
| ~$50/month for a big disk | ~$0.023/GB/month |

---

### 🔵 Q4 — What is IAM and why does it matter?

💡 **Think of IAM as the security guard of AWS.**

**IAM = Identity and Access Management.** It controls WHO can do WHAT on AWS.

| Concept | Meaning | Example in this project |
|---|---|---|
| User | An identity (person or app) | `finrag-app-user` |
| Access Key | Username for code | `AWS_ACCESS_KEY_ID` |
| Secret Key | Password for code | `AWS_SECRET_ACCESS_KEY` |
| Policy | Rules about what's allowed | "Can read S3, call Bedrock" |
| Role | Permissions attached to a service | `finrag-task-execution-role` |

> ⚠️ **Golden Rule:** Never commit keys to GitHub. GitHub's secret scanner auto-blocked our push when it detected a key in the code.

---

### 🔵 Q5 — What is an AWS Region and why does it matter?

💡 **Think of regions as Amazon's offices in different cities.**

| Region Code | Location |
|---|---|
| `us-east-1` | North Virginia, USA ← **we use this** |
| `ap-south-1` | Mumbai, India |
| `eu-west-1` | Ireland |
| `ap-southeast-1` | Singapore |

**Why `us-east-1`?**
- Bedrock's Nova Lite model is only available in certain regions
- All AWS resources (S3, EKS, ECS, ECR) must be in the same region to avoid latency and extra cost

---

## 🟣 AWS Bedrock & AI Models

---

### 🟣 Q6 — What is AWS Bedrock?

💡 **Think of Bedrock as an AI model vending machine.**

You send in text → Bedrock runs the AI model → you get back an answer → you pay per 1,000 tokens.

**Models available on Bedrock:**

| Model | Made by | Used in this project? |
|---|---|---|
| Nova Lite | Amazon | ✅ Yes — answers questions |
| Titan Embeddings | Amazon | ✅ Yes — converts text to vectors |
| Claude 3.5 Sonnet | Anthropic | ❌ Needs payment verification |
| Llama 3 | Meta | ❌ Not used |

```python
# From src/rag/bedrock_llm.py
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
response = bedrock.invoke_model(
    modelId="amazon.nova-lite-v1:0",
    body=json.dumps({"messages": [{"role": "user", "content": prompt}]})
)
```

---

### 🟣 Q7 — What is Amazon Nova Lite?

💡 **Think of Nova Lite as ChatGPT made by Amazon.**

| Property | Value |
|---|---|
| Model ID | `amazon.nova-lite-v1:0` |
| Response time | 2–7 seconds |
| Max context | 300K tokens |
| Cost | ~$0.06 per million input tokens |
| Access | Available immediately on free tier |

**Why not Claude?** Claude models on Bedrock require payment method verification. Nova Lite works immediately — that's why we chose it.

**What Nova Lite does in this project:**
1. Receives the retrieved document chunks (context)
2. Understands your question
3. Writes a grounded answer — only from the document, never guesses

---

### 🟣 Q8 — What are embeddings and what is Amazon Titan?

💡 **Think of embeddings as GPS coordinates for the meaning of text.**

A computer can't compare the "meaning" of two sentences. But it can compare numbers. **Embeddings = converting text into numbers that represent meaning.**

```
"Tesla revenue 2024"     → [0.23, -0.45, 0.78, ...]  ← 1536 numbers
"Tesla income this year" → [0.24, -0.44, 0.77, ...]  ← very similar ✅
"Pizza recipe"           → [0.91,  0.12, -0.33, ...]  ← very different ❌
```

```python
# From src/rag/embeddings.py
response = bedrock.invoke_model(
    modelId="amazon.titan-embed-text-v1",
    body=json.dumps({"inputText": chunk_text})
)
embedding = json.loads(response["body"].read())["embedding"]  # 1536 floats
```

Amazon Titan converts every PDF chunk into 1536 numbers. When you ask a question, it converts your question too — then finds chunks with the most similar numbers.

---

### 🟣 Q9 — What is throttling and how did we handle it?

💡 **Think of throttling as a speed limit on a highway.**

**Error we saw:**
```
ThrottlingException: Too many requests. Please try again later.
```

| Service | Free Tier Rate Limit |
|---|---|
| Titan Embeddings | ~5 requests/second |
| Nova Lite | ~10 requests/second |

**Impact:** 800 chunks ÷ 5 per second = 160 seconds = ~4 minutes to index one document.

**Fix — process in batches:**
```python
BATCH_SIZE = 5
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i + BATCH_SIZE]
    embeddings = [embed(chunk) for chunk in batch]
    time.sleep(0.2)  # pause between batches
```

**Long-term fix:** Request a quota increase from AWS Support — free, takes 1–2 days.

---

## 🟢 AWS Compute & Serverless

---

### 🟢 Q10 — What is AWS Lambda?

💡 **Think of Lambda as a motion-sensor light — only turns on when needed.**

| Feature | Lambda | Always-on Server |
|---|---|---|
| Cost when idle | $0.00 | Keeps charging |
| Setup | Upload code, done | Install OS, configure |
| Max runtime | 15 minutes | Unlimited |
| Best for | Short tasks, event-driven | APIs, long-running processes |

**In this project** — Lambda triggers when a PDF lands in S3:
```
PDF uploaded to S3 → S3 Event → Lambda triggers → Indexes PDF in background
```

Code lives in `src/lambda_handler/handler.py`. We used background threads for the demo, but Lambda is the production-grade approach.

---

### 🟢 Q11 — What is boto3?

💡 **Think of boto3 as the TV remote for AWS.**

`boto3` is the official Python library to interact with all AWS services.

```python
import boto3

session = boto3.Session(
    aws_access_key_id="AKIAXXX",
    aws_secret_access_key="secretXXX",
    region_name="us-east-1"
)

# Use S3
s3 = session.client("s3")
s3.put_object(Bucket="pritam-finrag-docs", Key="file.pdf", Body=data)

# Use Bedrock
bedrock = session.client("bedrock-runtime")
response = bedrock.invoke_model(modelId="amazon.nova-lite-v1:0", body=...)
```

Used across this project:
- `src/ingestion/s3_loader.py` — upload/download PDFs
- `src/rag/bedrock_llm.py` — call Nova Lite for answers
- `src/rag/embeddings.py` — call Titan for embeddings
- `src/ingestion/chart_extractor.py` — call Bedrock Vision for chart captions

---

### 🟢 Q12 — ECS vs EKS — what's the difference?

💡 **ECS = Amazon's simpler container runner. EKS = Full Kubernetes managed by Amazon.**

| Feature | ECS | EKS |
|---|---|---|
| Setup difficulty | Easy (10 min) | Hard (30–60 min) |
| Kubernetes | No — Amazon's own system | Yes — full K8s |
| Auto-scaling | Basic | Advanced (HPA) |
| Cost | Lower | Higher (control plane fee) |
| Best for | Simple apps | Complex microservices |
| Our live URL | http://13.222.137.204:8000 | http://finrag.44.206.217.242.nip.io |

**In this project we deployed to both** — ECS first (simpler), EKS second (production-grade).

---

## 🔴 Kubernetes (K8s) Fundamentals

---

### 🔴 Q13 — What is Kubernetes?

💡 **Think of Kubernetes as a smart manager for your app.**

Imagine you have a food delivery app:
- Normal day: 100 orders → 2 servers needed
- New Year's Eve: 10,000 orders → 20 servers needed
- One server crashes → traffic should automatically go to another

**Kubernetes handles all of this without you doing anything.**

| What K8s does | How it helps |
|---|---|
| Runs multiple copies of your app | No single point of failure |
| Restarts crashed containers | Always available |
| Scales up when traffic spikes | Never overloaded |
| Scales down when quiet | Saves money |
| Distributes traffic evenly | Fast responses |

---

### 🔴 Q14 — What is a Docker Container?

💡 **Think of a container as a lunchbox for your app.**

A container packages your app + Python + all libraries into one portable box. Anyone can run the same box and get the exact same app.

**Dockerfile in this project** (multi-stage — keeps the final image slim):
```dockerfile
# Stage 1: Install everything
FROM python:3.11-slim AS builder
RUN pip install fastapi uvicorn PyMuPDF boto3 sentence-transformers

# Stage 2: Copy only what's needed to run
FROM python:3.11-slim AS runtime
COPY --from=builder /usr/local/lib/python3.11/site-packages .
COPY src/ ./src/
USER finrag        # non-root user for security
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why multi-stage?** Builder stage can be 3 GB. Runtime stage is ~800 MB. Smaller = faster deploys + cheaper ECR storage.

---

### 🔴 Q15 — What is a Pod?

💡 **Think of a Pod as one delivery bike with one rider.**

A Pod is the smallest unit in Kubernetes — usually one running container.

- If the rider falls sick (container crashes) → Kubernetes automatically sends a new rider
- Pods are **temporary** — they can be killed and replaced anytime
- Each Pod gets its own internal IP address

```yaml
# k8s/deployment.yaml
containers:
- name: finrag-api
  image: 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest
  ports:
  - containerPort: 8000
  resources:
    requests:
      memory: "512Mi"
      cpu: "250m"
    limits:
      memory: "1Gi"
      cpu: "500m"
```

---

### 🔴 Q16 — What is a Deployment?

💡 **Think of a Deployment as a recipe: "always keep 2 copies of my app running."**

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finrag-api
  namespace: finrag
spec:
  replicas: 2          # keep 2 copies running always
  selector:
    matchLabels:
      app: finrag-api
  template:
    spec:
      containers:
      - name: finrag-api
        image: 020262236277.dkr.ecr.us-east-1.amazonaws.com/finrag-api:latest
```

**Rolling updates:** When you push a new image, Kubernetes replaces Pods one at a time — the app never goes down during a deploy.

---

### 🔴 Q17 — What is a Service?

💡 **Think of a Service as a hotel receptionist — one number, routes to whoever is free.**

Pods come and go. A Service provides a **stable address** that always points to healthy Pods.

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: finrag-service
  namespace: finrag
spec:
  selector:
    app: finrag-api        # routes to all Pods with this label
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer       # creates a public IP
```

| Service Type | What it does |
|---|---|
| `ClusterIP` | Internal only — Pods talk to each other |
| `NodePort` | Opens a port on every node |
| `LoadBalancer` | Creates a public IP ← **we use this** |

---

## 🟠 Kubernetes Advanced

---

### 🟠 Q18 — What is HPA (Horizontal Pod Autoscaler)?

💡 **Think of HPA as a restaurant calling in extra waiters when it gets busy.**

```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: finrag-hpa
  namespace: finrag
spec:
  scaleTargetRef:
    kind: Deployment
    name: finrag-api
  minReplicas: 1          # always at least 1 Pod
  maxReplicas: 5          # never more than 5 Pods
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70   # scale up if CPU > 70%
```

**Flow:**
```
Traffic spikes → CPU > 70% → HPA adds Pods  → CPU drops  ✅
Traffic drops  → CPU < 70% → HPA removes Pods → Cost drops ✅
```

---

### 🟠 Q19 — What is Amazon EKS?

💡 **Think of EKS as Kubernetes with Amazon as your IT department.**

Running Kubernetes yourself = managing the control plane (complex, risky). EKS = AWS manages the control plane, you just manage your apps.

**How we set up EKS in this project:**
```bash
eksctl create cluster \
  --name finrag-cluster-eks \
  --region us-east-1 \
  --nodegroup-name finrag-nodes \
  --node-type t3.small \
  --nodes 2 \
  --managed

kubectl apply -f k8s/
kubectl get pods -n finrag
```

**Lessons from our EKS setup:**

| Problem we hit | Fix |
|---|---|
| `AL2_x86_64` AMI not supported for K8s 1.34 | Use `AL2023_x86_64_STANDARD` |
| Wrong VPC subnets | Query EKS cluster VPC first, use its subnets |
| Two clusters created by accident | Delete duplicate with `eksctl delete cluster` |

---

### 🟠 Q20 — What is kubectl?

💡 **Think of kubectl as the steering wheel for Kubernetes.**

```bash
# See what's running
kubectl get pods -n finrag
kubectl get deployments -n finrag
kubectl get services -n finrag

# Read logs
kubectl logs deployment/finrag-api -n finrag --tail=50

# Restart (pull latest image)
kubectl rollout restart deployment/finrag-api -n finrag
kubectl rollout status deployment/finrag-api -n finrag

# Debug a pod
kubectl describe pod <pod-name> -n finrag

# Apply config changes
kubectl apply -f k8s/
```

---

### 🟠 Q21 — What is an Ingress?

💡 **Think of Ingress as a smart traffic director at the building entrance.**

A LoadBalancer Service = one public IP per service (expensive if you have many). Ingress = one IP, routes based on hostname or URL path.

```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: finrag-ingress
  namespace: finrag
spec:
  rules:
  - host: finrag.44.206.217.242.nip.io
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: finrag-service
            port:
              number: 80
```

> 💡 **nip.io trick:** `finrag.44.206.217.242.nip.io` automatically resolves to `44.206.217.242` — free DNS without buying a domain!

---

## 🔷 How It All Connects in This Project

---

### 🔷 Q22 — Walk me through the full architecture end-to-end

```
👤 User (Browser)
        │
        ▼
🌐 http://finrag.44.206.217.242.nip.io   ← EKS Ingress (nip.io free DNS)
        │
        ▼
⚙️  FastAPI App  ← Kubernetes Pod on EKS (auto-restarts, scales with HPA)
        │
   ┌────┴─────────────┐
   ▼                  ▼
🗄️ S3 (PDFs)      🤖 AWS Bedrock
pritam-finrag-docs    ├── Nova Lite       (generates answers)
        │             └── Titan Embeddings (text → vectors)
        ▼
🔄 Lambda  ← triggers on S3 upload, indexes PDF in background
```

**Full user flow:**
1. Open `http://finrag.44.206.217.242.nip.io`
2. Upload PDF → FastAPI saves to S3 → Lambda triggers → chunks + embeds → stores index
3. Ask question → encode with Titan → hybrid search (BM25 + dense) → rerank → send top chunks to Nova Lite → get answer
4. Kubernetes keeps everything running 24/7, restarts crashes, scales under load

---

### 🔷 Key Terms Cheat Sheet

| Term | Simple meaning |
|------|---------------|
| **AWS** | Amazon's rental shop for computers and internet tools |
| **S3** | Amazon's Google Drive for developers (stores PDFs) |
| **IAM** | Security guard — controls who can do what |
| **Region** | Which city's data center to use (`us-east-1` = Virginia) |
| **Bedrock** | Amazon's AI model vending machine |
| **Nova Lite** | Amazon's own LLM — fast and cheap for document Q&A |
| **Titan Embeddings** | Converts text to 1536-dimensional vectors for search |
| **Throttling** | "Too many requests — slow down!" rate limit error |
| **Lambda** | Run code only when triggered — zero idle cost |
| **boto3** | Python remote control for all AWS services |
| **ECR** | Amazon's Docker image registry (like Docker Hub) |
| **ECS** | Amazon's simple container runner |
| **EKS** | Amazon's managed Kubernetes |
| **Docker** | Tool to package your app into a portable container |
| **Container** | Box with your app + everything it needs to run |
| **Pod** | One running container in Kubernetes |
| **Deployment** | Recipe: "keep N copies of my app running always" |
| **Service** | Stable address that routes traffic to healthy Pods |
| **HPA** | Auto-adds Pods when busy, removes when quiet |
| **kubectl** | Command-line steering wheel for Kubernetes |
| **Ingress** | Smart traffic director — routes URLs to services |
| **nip.io** | Free DNS trick: `finrag.1.2.3.4.nip.io` resolves to `1.2.3.4` |
