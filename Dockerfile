# Use uv for faster Python package installation
FROM python:3.12-slim AS builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Copy pyproject.toml
COPY pyproject.toml README.md ./

# Create virtual environment and install dependencies with uv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.0.0" "torchvision>=0.15.0" && \
    uv pip install --no-cache .

# Final stage
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=2 \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    ffmpeg \
    libsndfile1 \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 --gid 1001 appuser

WORKDIR /code
COPY --from=builder /opt/venv /opt/venv
COPY . .

RUN mkdir -p /code/uploads /code/models /code/logs && \
    chown -R appuser:appgroup /code

USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
