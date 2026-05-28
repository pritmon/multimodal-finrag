# Lambda container image for the FinRAG document processor
# Uses the AWS-provided Python 3.11 Lambda base image.

FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies for PyMuPDF and Pillow
RUN dnf install -y \
    gcc \
    gcc-c++ \
    make \
    libffi-devel \
    openssl-devel \
    mupdf \
    mupdf-devel \
    && dnf clean all

# Set working directory to Lambda task root
WORKDIR ${LAMBDA_TASK_ROOT}

# Copy requirements first for layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        # Core dependencies (subset for Lambda — skip heavy training deps)
        llama-index>=0.10.0 \
        llama-index-llms-bedrock \
        llama-index-embeddings-bedrock \
        boto3>=1.34.0 \
        PyMuPDF>=1.23.0 \
        Pillow>=10.2.0 \
        sentence-transformers>=2.5.0 \
        rank_bm25>=0.2.2 \
        pydantic-settings>=2.2.0 \
        tenacity>=8.2.0 \
        open-clip-torch>=2.24.0 \
        # Exclude heavy deps not needed in Lambda
        # (torch is bundled via Lambda layers or container separately)
    && true

# Install PyTorch CPU-only (reduces image size; GPU not available in Lambda)
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ ./src/

# Pre-download CLIP model weights so they're baked into the image
RUN python -c "\
import open_clip; \
open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai'); \
print('CLIP weights cached')"

# Lambda handler entry point: module.function
CMD ["src.lambda_handler.handler.handler"]
