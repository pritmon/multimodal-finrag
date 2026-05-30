# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make libffi-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install all deps in one shot using exact versions from working environment
RUN pip install --no-cache-dir \
    fastapi==0.128.8 \
    "uvicorn[standard]==0.39.0" \
    pydantic==2.13.4 \
    pydantic-settings==2.11.0 \
    python-multipart==0.0.20 \
    python-dotenv==1.2.1 \
    httpx==0.28.1 \
    aiofiles==25.1.0 \
    tenacity==8.5.0 \
    boto3==1.40.61 \
    PyMuPDF==1.26.5 \
    Pillow==11.3.0 \
    rank_bm25==0.2.2 \
    numpy==1.26.4 \
    pandas==2.3.3 \
    tqdm==4.67.3

# PyTorch CPU
RUN pip install --no-cache-dir \
    torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ML
RUN pip install --no-cache-dir \
    transformers==4.41.0 \
    sentence-transformers==3.0.0 \
    peft==0.11.0 \
    accelerate==0.30.0

# CLIP
RUN pip install --no-cache-dir open-clip-torch==2.24.0

# LlamaIndex — install core + only needed plugins (skip meta-package)
RUN pip install --no-cache-dir llama-index-core==0.10.68
RUN pip install --no-cache-dir \
    llama-index-llms-bedrock==0.4.2 \
    llama-index-embeddings-bedrock==0.7.4

# Pre-download model weights into the image
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Sentence models cached')"

RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); \
print('CLIP cached')"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN groupadd -r finrag && useradd -r -g finrag finrag

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /home/finrag/.cache

COPY src/ ./src/

RUN mkdir -p /app/index_store /app/models && \
    chown -R finrag:finrag /app

USER finrag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
