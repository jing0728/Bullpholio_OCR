# ── Bullpholio API — Dockerfile ──────────────────────────────────────────────
#
# Multi-stage build:
#   builder  — install Python deps into a venv
#   runtime  — copy only the venv + source; no build tools in the final image
#
# Build:
#   docker build -t bullpholio-api .
#
# Run:
#   docker run -p 8000:8000 bullpholio-api
#
# Run with OCR support and custom settings:
#   docker run -p 8000:8000 \
#     -e MAX_FILE_SIZE_MB=100 \
#     -e LOG_LEVEL=DEBUG \
#     -e CORS_ORIGINS="http://localhost:8080,https://app.example.com" \
#     bullpholio-api
#
# Health check:
#   curl http://localhost:8000/health
#
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some Python packages (OpenCV, Pillow, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir fastapi uvicorn[standard]


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Runtime system libraries only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy source
COPY bullpholio/ ./bullpholio/
COPY api/        ./api/

# Non-root user for security
RUN useradd -m -u 1000 appuser
USER appuser

# Expose port
EXPOSE 8000

# Environment variable defaults (override at runtime)
ENV MAX_FILE_SIZE_MB=50
ENV LOG_LEVEL=INFO
ENV CORS_ORIGINS=*

# Health check (Docker / Kubernetes readiness)
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
