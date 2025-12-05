FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pydantic-settings

# Copy application files
COPY api_server.py .
COPY ollama_chroma.py .
COPY bulk_ingest.py .

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db

# Expose port
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV OLLAMA_URL=http://ollama:11434
ENV EMBED_MODEL=embeddinggemma
ENV CHROMA_PERSIST_DIR=/app/chroma_db
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the API server
CMD ["python", "api_server.py"]

