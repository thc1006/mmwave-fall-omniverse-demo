# Multi-stage build for mmWave Fall Detection API
# Stage 1: Build dependencies
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
# Use CPU-only torch to reduce image size (remove --index-url for GPU support)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    numpy \
    pydantic

# Stage 2: Production image
FROM python:3.11-slim AS production

WORKDIR /app

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY ml/fallnet_model.py /app/ml/fallnet_model.py
COPY services/api/ /app/services/api/

# Create models directory for volume mount
RUN mkdir -p /app/models && chown -R appuser:appuser /app

# Copy default model (can be overridden by volume mount)
COPY ml/fallnet_lstm.pt /app/models/fallnet.pt

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/fallnet.pt

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the API server
CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
