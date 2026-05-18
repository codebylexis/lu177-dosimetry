# Monte Carlo Dosimetry API — Docker image
# =========================================
# Multi-stage build: keeps the final image lean by separating
# dependency installation from the runtime layer.
#
# Build:   docker build -t mc-dosimetry .
# Run:     docker run -p 8000:8000 mc-dosimetry
# Dev:     docker run -p 8000:8000 -v $(pwd)/src:/app/src mc-dosimetry

# ---------------------------------------------------------------------------
# Stage 1: dependency builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools (needed for some scipy/numpy wheels on ARM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Lexi Sierfeld <sierfeld@sas.upenn.edu>"
LABEL description="Monte Carlo dosimetry API for Lu-177 DOTATATE"
LABEL version="1.0.0"

# Non-root user for security
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
COPY src/   ./src/
COPY api/   ./api/

# Set PYTHONPATH so src/ modules are importable
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Tune for container environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Limit OpenBLAS threads to avoid oversubscription in containers
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

USER appuser

EXPOSE 8000

# Health check — polls the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]
