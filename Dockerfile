# ─── Stage 1: Python ML Pipeline ─────────────────────────────────────────────
FROM python:3.11-slim AS ml-pipeline

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY api/ api/
COPY dataset/ dataset/
COPY run_pipeline.py .

# Create output directories
RUN mkdir -p outputs/reports outputs/models outputs/submissions outputs/shap

# Default: run the full pipeline
CMD ["python", "run_pipeline.py"]

# ─── Stage 2: API Server ────────────────────────────────────────────────────
FROM ml-pipeline AS api-server

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
