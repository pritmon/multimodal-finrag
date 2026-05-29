# AWS, Bedrock & Kubernetes — Explained Simply
### For someone completely new 🌱

---

## PART 1 — AWS (Amazon Web Services)

---

### What is AWS?

Imagine you want to open a restaurant. You need:
- A kitchen (server/computer)
- A fridge (storage)
- A phone line (networking)
- A billing system

You could buy all this yourself. Very expensive. Or you could **rent** it from someone who already has everything built.

**AWS = Amazon's rental shop for computers, storage, and internet tools.**

You pay only for what you use. No upfront cost. Netflix, Airbnb, NASA — they all use AWS.

---

### AWS Free Tier — What is it?

When you create an AWS account, Amazon gives you **free usage** for 12 months on many services. Like a free trial.

In this project we use:
- **S3** — 5GB free storage
- **Bedrock** — limited free calls per month
- **EC2** — 750 hours/month of a small computer

After free tier ends, you pay. That's why we got throttled — too many requests on the free plan.

---

### What is S3?

**S3 = Simple Storage Service**

It's like Google Drive but for developers. You store files (called "objects") in "buckets" (like folders).

In this project:
- Bucket name: `pritam-finrag-docs`
- When you upload a PDF, it goes to S3 first
- Key (file path): `documents/{job_id}/infosys_annual_report_2025.pdf`

```python
s3.put_object(
    Bucket="pritam-finrag-docs",
    Key="documents/abc123/infosys.pdf",
    Body=pdf_bytes
)
```

**Why S3?** Durable (99.999999999% uptime), cheap (~$0.023/GB/month), accessible from anywhere.

---

### What is IAM?

**IAM = Identity and Access Management**

It's the security guard of AWS. Controls WHO can do WHAT.

- **User** = A person or app with an identity
- **Access Key** = Like a username + password for code (not humans)
- **Policy** = Rules about what the user can do

In this project we use:
- `AWS_ACCESS_KEY_ID` = who we are
- `AWS_SECRET_ACCESS_KEY` = our password
- These give access to S3 and Bedrock

**Golden rule:** Never commit these to GitHub. We learned this the hard way — GitHub blocked our push because it detected the keys.

---

### What is a Region?

AWS has data centers all over the world — called **regions**.

- `us-east-1` = North Virginia, USA (what we use)
- `ap-south-1` = Mumbai, India
- `eu-west-1` = Ireland

You pick the region closest to your users for fastest speed. Bedrock's Nova Lite model is only available in certain regions — that's why we use `us-east-1`.

---

## PART 2 — AWS Bedrock

---

### What is AWS Bedrock?

Bedrock is AWS's service for using AI models without building them yourself.

Think of it like a **AI model vending machine**:
- You put in text (your question)
- It runs a powerful AI model
- You get back an answer
- You pay per 1000 words processed

**Models available on Bedrock:**
- Amazon Nova (what we use — works immediately)
- Anthropic Claude (blocked for us — needs payment verification)
- Meta Llama
- Mistral
- Stability AI (for images)

---

### What is Amazon Nova Lite?

Nova Lite is Amazon's own AI model. Like ChatGPT but made by Amazon.

- **Fast** — responds in 2-7 seconds
- **Cheap** — costs very little per request
- **Good enough** — handles Q&A on documents well

We use it to:
1. Read the retrieved document chunks
2. Understand the question
3. Write a clear answer

---

### What is Amazon Titan Embeddings?

"Embedding" = converting text into a list of 1536 numbers that represent its meaning.

**Why numbers?** Computers can't compare meanings of words directly. But they can compare numbers. Two sentences with similar meanings will have similar numbers.

Example:
- "Tesla revenue" → [0.23, -0.45, 0.78, ...] (1536 numbers)
- "Tesla income" → [0.24, -0.44, 0.77, ...] (very similar numbers!)
- "Pizza recipe" → [0.91, 0.12, -0.33, ...] (very different numbers)

Amazon Titan converts every chunk of your PDF into these numbers. Then when you ask a question, it converts your question too, and finds the chunks with the most similar numbers.

---

### What is Bedrock's Rate Limit (Throttling)?

AWS limits how many requests you can make per second. Like a highway speed limit.

**Free tier limit for Titan Embeddings:** ~5 requests/second

That's why indexing takes ~4 minutes for 800 chunks:
- 800 chunks ÷ 5 per second = 160 seconds minimum
- Plus processing time = ~4 minutes total

**Error we saw:** `ThrottlingException: Too many requests`

**Fix:** Send max 5 requests at a time, pause 0.2s between batches.

To get faster: Request a quota increase from AWS Support (free, takes 1-2 days).

---

### What is AWS Lambda?

Lambda = Run code without managing a server.

Normal server: You rent a computer 24/7. Costs money even when idle.

Lambda: Your code only runs when triggered. Pay only for the seconds it runs.

**In this project:** Lambda is set up as an alternative for background indexing. When a PDF is uploaded to S3, Lambda automatically triggers and indexes it. Like a motion sensor light — only turns on when needed.

We didn't fully use Lambda in our demo (used background threads instead), but the code is written in `src/lambda_handler/handler.py`.

---

### What is boto3?

`boto3` = The Python library to talk to AWS services.

Like a TV remote — you press buttons (call functions) and it sends signals (API calls) to AWS.

```python
import boto3

# Create a session with your credentials
session = boto3.Session(
    aws_access_key_id="AKIAXXX",
    aws_secret_access_key="secretXXX",
    region_name="us-east-1"
)

# Use S3
s3 = session.client("s3")
s3.put_object(Bucket="my-bucket", Key="file.pdf", Body=data)

# Use Bedrock
bedrock = session.client("bedrock-runtime")
response = bedrock.invoke_model(modelId="amazon.nova-lite-v1:0", body=...)
```

---

## PART 3 — Kubernetes (K8s)

---

### What is Kubernetes?

Imagine you have a food delivery app. On a normal day, 100 people order. On New Year's Eve, 10,000 people order. You need:
- More delivery bikes (servers) on busy days
- Fewer on quiet days
- If one bike breaks, send another automatically

**Kubernetes = The manager that handles all of this automatically.**

It runs your app in "containers" (boxes) and manages:
- How many boxes are running
- Restarting boxes that crash
- Sharing traffic between boxes
- Scaling up/down based on demand

---

### What is a Container?

A container = A box that has your app + everything it needs to run.

Think of it like a **lunchbox**:
- Regular food: your app code
- Lunchbox: container (includes Python, libraries, everything)
- Anyone can open the same lunchbox and get the same food

**Docker** makes the lunchbox. **Kubernetes** manages thousands of lunchboxes.

In this project, you'd package the FastAPI app in a Docker container:
```dockerfile
FROM python:3.9
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0"]
```

---

### What is a Pod?

A Pod = The smallest unit in Kubernetes. Usually one container.

Like one delivery bike with one rider.

If the rider falls sick (container crashes), Kubernetes automatically sends a new rider.

---

### What is a Deployment?

A Deployment = Instructions for Kubernetes about how many Pods to run.

```yaml
# Tell Kubernetes: run 3 copies of our FinRAG app
apiVersion: apps/v1
kind: Deployment
metadata:
  name: finrag-api
spec:
  replicas: 3        # 3 copies running at once
  template:
    spec:
      containers:
      - name: finrag
        image: finrag:latest
        ports:
        - containerPort: 8000
```

If any of the 3 copies crash, Kubernetes restarts them automatically.

---

### What is a Service?

A Service = A single address that sends traffic to all your Pods.

Like a receptionist at a hotel. You call the hotel number (one number), the receptionist routes you to whichever staff member is free.

```yaml
apiVersion: v1
kind: Service
metadata:
  name: finrag-service
spec:
  selector:
    app: finrag
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer   # gives a public IP
```

---

### What is Horizontal Pod Autoscaling (HPA)?

HPA = Kubernetes automatically adds or removes Pods based on traffic.

Like a restaurant that calls in extra waiters when it gets busy, and sends them home when quiet.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: finrag-hpa
spec:
  scaleTargetRef:
    name: finrag-api
  minReplicas: 1      # minimum 1 Pod always running
  maxReplicas: 10     # maximum 10 Pods when very busy
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70   # scale up if CPU > 70%
```

---

### What is Amazon EKS?

**EKS = Elastic Kubernetes Service**

Running Kubernetes yourself is hard — lots of setup. AWS EKS is a managed version where Amazon handles the Kubernetes setup for you.

You just:
1. Create an EKS cluster (one command)
2. Deploy your app
3. AWS handles the rest

In this project, the `k8s/` folder has deployment files ready for EKS.

---

### Kubernetes vs Lambda — which to use?

| | Lambda | Kubernetes (EKS) |
|--|--------|-----------------|
| **Best for** | Short tasks, event-driven | Long-running apps, APIs |
| **Cost** | Pay per request | Pay per hour (even when idle) |
| **Scaling** | Automatic | Automatic with HPA |
| **Setup** | Very easy | Complex |
| **Our app** | Good for indexing PDFs | Good for the API server |

**In this project:**
- Lambda → triggered when PDF uploaded to S3, runs indexing
- EKS → runs the FastAPI server 24/7

---

## PART 4 — How it all connects in this project

```
User (Browser)
    ↓
FastAPI Server (running in Kubernetes Pod on EKS)
    ↓           ↓
   S3          Bedrock
(store PDFs)  (Nova Lite LLM + Titan Embeddings)
    ↓
  Lambda
(index PDF in background when uploaded to S3)
```

**Full flow:**
1. You open `localhost:8000` (or a public URL if deployed on EKS)
2. Upload PDF → FastAPI saves it to S3 → Lambda triggers → indexes it
3. Ask question → FastAPI retrieves chunks from index → sends to Bedrock Nova Lite → get answer
4. Kubernetes keeps the FastAPI server always running, restarts if it crashes, scales if more users come

---

## Key Terms Cheat Sheet

| Term | Simple meaning |
|------|---------------|
| AWS | Amazon's rental shop for computers and internet tools |
| S3 | Amazon's Google Drive for developers |
| IAM | Security guard — controls who can do what |
| Region | Which city's data center to use |
| Bedrock | Amazon's AI model vending machine |
| Nova Lite | Amazon's own AI model (like ChatGPT) |
| Titan | Amazon's text-to-numbers converter |
| Throttling | "Too many requests, slow down!" |
| Lambda | Run code only when needed, no server required |
| boto3 | Python remote control for AWS |
| Docker | Lunchbox for your app |
| Container | Box with your app + everything it needs |
| Pod | One running container in Kubernetes |
| Deployment | Instructions: "run 3 copies of my app" |
| Service | Receptionist that routes traffic to Pods |
| HPA | Auto-adds more Pods when busy |
| EKS | Amazon's managed Kubernetes |
| kubectl | Command line tool to control Kubernetes |
