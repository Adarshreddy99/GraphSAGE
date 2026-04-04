# 1. Builder Stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies (for things like Faiss or specialized PyTorch wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements to the builder (caches them in a layer)
COPY requirements.txt .

# Install the "ML Foundation" separately first to ensure stability and proper CPU wheels
RUN pip install --user --no-cache-dir torch==2.2.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --user --no-cache-dir torch-geometric==2.5.2 \
    torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cpu.html

# Install the rest of the application dependencies
RUN pip install --user --no-cache-dir -r requirements.txt

# 2. Runtime Stage
FROM python:3.11-slim

WORKDIR /app

# Install dos2unix for line ending stability and build-essential for any dynamic needs
RUN apt-get update && apt-get install -y --no-install-recommends \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Copy only the installed packages from the builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src:/app

# Copy the Source Code and Parameters
COPY src/ ./src/
COPY params.yaml .
COPY scripts/start.sh ./scripts/start.sh

# Scrub line endings and make executable
RUN dos2unix ./scripts/start.sh && chmod +x ./scripts/start.sh

# Expose ports: 8000 (FastAPI) and 8080 (Gradio)
EXPOSE 8000
EXPOSE 8080

# Use 'sh' as a shell wrapper to ensure compatibility
CMD ["sh", "./scripts/start.sh"]
