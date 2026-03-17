# ── Stage 1: deps (only rebuilds when requirements.txt changes) ──────────────
FROM python:3.11-slim AS deps

# System packages needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first — Docker caches this layer until it changes
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: final image ──────────────────────────────────────────────────────
FROM deps AS final

# Copy application code (code changes hit only this layer, not pip)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY fonts/ ./fonts/

# Create upload/output dirs
RUN mkdir -p uploads outputs

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
