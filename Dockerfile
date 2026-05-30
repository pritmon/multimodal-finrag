# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for PyMuPDF, Pillow, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ make libffi-dev libssl-dev \
    libmupdf-dev mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download sentence-transformers and cross-encoder model weights
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached')"

# Pre-download CLIP weights
RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); \
print('CLIP cached')"

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Non-root user for security
RUN groupadd -r finrag && useradd -r -g finrag finrag

# Copy installed packages and cached models from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache /home/finrag/.cache

# Runtime system libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy source code
COPY src/ ./src/

# Index store volume mount point
RUN mkdir -p /app/index_store /app/models && \
    chown -R finrag:finrag /app

USER finrag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
