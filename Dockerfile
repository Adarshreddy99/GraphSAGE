# 1. Builder Stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies (for things like Faiss or specialized PyTorch wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements to the builder (caches them in a layer)
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# 2. Runtime Stage
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed packages from the builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy the Source Code and Parameters
COPY src/ ./src/
COPY params.yaml .
COPY scripts/start.sh ./scripts/start.sh

# Ensure the startup script is executable
RUN chmod +x ./scripts/start.sh

# Expose ports: 8000 (FastAPI) and 8080 (Gradio)
EXPOSE 8000
EXPOSE 8080

# The Entrypoint: Our startup orchestrator
CMD ["./scripts/start.sh"]
